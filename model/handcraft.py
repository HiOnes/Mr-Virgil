import torch
import torch.nn.functional as F
from torch import nn, Tensor
from copy import deepcopy
from typing import Optional, List, Tuple

class GAT_TRANSFORMER(nn.Module):
    def __init__(self, args, device):
        
        super().__init__()        

        self.dot_attn = Dot_Attn(device).to(device)
        self.others_num = args.swarm_num - 1
        self.max_cam_num = args.max_cam_num
        self.dim = args.embed_size
        self.device = device
        self.max_cov = 10.0

    def forward(self, others_feat: Tensor, others_cam: Tensor):
        '''
        input:
            others_feat: torch.Tensor, [bs*(n+1)*n, 7+1]
                @@@ note: bs: batch_size || n+1: num of robots || n: num of other robots || 7+1: [x y z qx qy qz qw dis] @@@
            others_cam: torch.Tensor, [bs*(n+1)*m, 3]
                @@@ note: bs: batch_size || n+1: num of robots || m: num of max camera observations || 3: bearing [bx, by, bz] @@@
        output: 
            outputs: dict
        '''
        n, m = self.others_num, self.max_cam_num
        bsn = int(others_feat.shape[0] / n) # bsn = batchsize * (n+1)
        others_prior_pos = others_feat[:, :3] # [bsn*n, 3]
        others_prior_pos = others_prior_pos.reshape(bsn, n, 3) # [bsn, n, 3]
        others_prior_dir = F.normalize(others_prior_pos, p=2.0, dim=-1) # [bsn, n, 3]

        others_cam = others_cam.reshape(bsn, -1, others_cam.shape[1]) # [bsn, m, 3]
        cam_norm2 = torch.norm(others_cam, p=2, dim=2)
        cam_lost_mask = cam_norm2 < 1e-4 # [bsn, m]

        cam_lost_mask_split = cam_lost_mask.reshape(-1, n+1, m) # [bs, n+1, m]
        others_prior_dir_split = others_prior_dir.reshape(-1, n+1, n, 3) # [bs, n+1, n, 3]
        others_cam_split = others_cam.reshape(-1, n+1, m, 3) # [bs, n+1, m, 3]
        others_feat_split = others_feat.reshape(-1, n+1, n, others_feat.shape[-1]) # [bs, n+1, n, 7+1]
        out_pos = torch.zeros(others_prior_dir_split.shape).to(self.device)
        out_cov = torch.zeros(out_pos.shape[0], n+1, n, 1).to(self.device)
        out_prob = torch.zeros(out_pos.shape[0], n+1, n, m).to(self.device)
        out_scores = torch.zeros(out_pos.shape[0], n+1, n+1, m+1).to(self.device)
        out_indices = torch.zeros(out_pos.shape[0], n+1, n, 1).to(self.device)

        for k in range(n+1):
            others_prior_dir_k = others_prior_dir_split[:, k, :, :] # [bs, n, 3]
            others_cam_k = others_cam_split[:, k, :, :] # [bs, m, 3]
            cam_lost_mask_k = cam_lost_mask_split[:, k, :] # [bs, m]
            others_feat_k = others_feat_split[:, k, :, :] # [bs, n, 7+1]
            dis_k = others_feat_k[:,:,-1] # [bs, n]
            dis_lost_mask_k = dis_k < 1e-4 # [bs, n]
            if (cam_lost_mask_k.all() or dis_lost_mask_k.any()):  # all cams are lost or any distance is lost
                # print('all cams are lost or any distance is lost')
                prob_k = torch.zeros(others_prior_dir_k.shape[0], n, m).to(self.device)
                cov_k = torch.ones(others_prior_dir_k.shape[0], n, 1).to(self.device) * self.max_cov
                pos_k = others_feat_k[:,:,:3] # [bs, n, 3]
                scores_k = -torch.inf * torch.ones(others_prior_dir_k.shape[0], n+1, m+1).to(self.device)
                indices_k = -torch.ones(others_prior_dir_k.shape[0], n, 1).to(self.device).to(torch.int64)
            else:
                match_k = self.cos_match(others_prior_dir_k, others_cam_k, cam_lost_mask_k)
                pos_k = dis_k.unsqueeze(2) * match_k['cam'] # [bs, n, 3]
                scores_k = -torch.inf * torch.ones(others_prior_dir_k.shape[0], n+1, m+1).to(self.device)
                prob_k, cov_k, indices_k = match_k['prob'], match_k['cov'], match_k['indices']

            out_pos[:, k, :, :] = pos_k
            out_cov[:, k, :, :] = cov_k
            out_prob[:, k, :, :] = prob_k
            out_scores[:, k, :, :] = scores_k
            out_indices[:, k, :, :] = indices_k
        
        out_pos = out_pos.flatten(0, 1) # [bsn, n, 3]
        out_cov = out_cov.flatten(0, 1) # [bsn, n, 1]
        out_prob = out_prob.flatten(0, 1) # [bsn, n, m]
        out_scores = out_scores.flatten(0, 1) # [bsn, n+1, m+1]
        out_indices = out_indices.flatten(0, 1) # [bsn, n, 1]

        # if not valid, modify cov and pos
        valid = out_indices > -1
        invalid_cov = torch.tensor(self.max_cov, dtype=out_cov.dtype, device=out_cov.device)
        out_cov = torch.where(valid, out_cov, invalid_cov)
        out_pos = torch.where(valid, out_pos, others_prior_pos)

        outputs = {'prob': out_prob, 'pos': out_pos, 'cov': out_cov, 'scores': out_scores, 'indices': out_indices}

        return outputs

    def cos_match(self, prior_dir: Tensor, cam_dir: Tensor, cam_lost_mask: Tensor):
        '''
        input:
            prior_dir: torch.Tensor, [bsn, n, 3]
            cam_dir: torch.Tensor, [bsn, m, 3]
            cam_lost_mask: torch.Tensor, [bsn, m]
        output:
            (match_cam, prob, match_cos_similarity, var, indices, cov): tuple
        '''
        n, m = prior_dir.shape[1], cam_dir.shape[1]
        cos_similarity, prob = self.dot_attn(prior_dir, cam_dir, key_padding_mask=cam_lost_mask) # prob:[bsn, n, m]
        indices = torch.argmax(prob, dim=-1, keepdim=True) # [bsn, n, 1]
        match_cam_index = indices.repeat(1, 1, 3) # [bsn, n, 3]
        match_cam = torch.gather(cam_dir, dim=1, index=match_cam_index) # [bsn, n, 3]
        match_cos_similarity = torch.gather(cos_similarity, dim=2, index=indices) # [bsn, n, 1]
        cov = (1 - match_cos_similarity) * 100.0 # [bsn, n, 1]
        cov = torch.clamp(cov, 0.01, self.max_cov)
        match_valid = match_cos_similarity > 0.99 # [bsn, n, 1]

        attn_dir = torch.bmm(prob, cam_dir) # [bsn, n, 3]
        attn_dir = F.normalize(attn_dir, p=2.0, dim=2) # [bsn, n, 3]
        var = torch.zeros(attn_dir.shape[0], n, 1).to(self.device) # [bsn, n, 1]
        for i in range(n):
            mean = attn_dir[:, i, :].unsqueeze(1) # [bsn, 1, 3]
            gap_mean = cam_dir - mean # [bsn, m, 3]
            gap_square_mean = torch.norm(gap_mean, p=2.0, dim=2, keepdim=True) # [bsn, m, 1]
            bmm = torch.bmm(prob[:,i,:].unsqueeze(1), gap_square_mean) # [bsn, 1, 1]
            var[:, i, :] = bmm.squeeze(1) 
        
        invalid_index = torch.tensor(-1, dtype=indices.dtype, device=indices.device)
        invalid_cov = torch.tensor(self.max_cov, dtype=cov.dtype, device=cov.device)
        indices = torch.where(match_valid, indices, invalid_index)
        cov = torch.where(match_valid, cov, invalid_cov)
        out_match = { 'cam': match_cam, 'prob': prob, 'cos_similarity': match_cos_similarity, 'var': var, 'indices': indices, 'cov': cov }
        return out_match

class Dot_Attn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    def forward(self, query: Tensor, key: Tensor, key_padding_mask: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        '''
        input:
            query: torch.Tensor, [bsn, n, dim]
            key: torch.Tensor, [bsn, m, dim]
            key_padding_mask: Optional[torch.Tensor], [bsn, m]
        output:
            cos_similarity: torch.Tensor, [bsn, n, m]
            attn_weight: torch.Tensor, [bsn, n, m]
        '''
        if key_padding_mask is not None:
            mask = key_padding_mask.float() # [bsn, m]
            mask = mask.masked_fill(key_padding_mask, float('-inf')).unsqueeze(1) # [bsn, 1, m]
            cos_similarity = torch.baddbmm(mask, query, key.transpose(-2, -1)) # [bsn, n, m]
        else:
            cos_similarity = torch.bmm(query, key.transpose(-2, -1)) # [bsn, n, m]
        attn_weight = F.softmax(cos_similarity, dim=-1) # [bsn, n, m]
        return cos_similarity, attn_weight