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
            others_feat: torch.Tensor, [bs*1*n, 7+1]
            others_cam: torch.Tensor, [bs*1*m, 3]
        output:
            (prob, pos, cov, scores, indices): tuple
        '''
        n, m = self.others_num, self.max_cam_num
        bsn = int(others_feat.shape[0] / n) # bsn = batchsize = 1
        others_prior_pos = others_feat[:, :3] # [bsn*n, 3]
        others_prior_pos = others_prior_pos.reshape(bsn, n, 3) # [bsn, n, 3]
        others_prior_dir = F.normalize(others_prior_pos, p=2.0, dim=-1) # [bsn, n, 3]

        others_cam = others_cam.reshape(bsn, m, others_cam.shape[1]) # [bsn, m, 3]
        cam_norm2 = torch.norm(others_cam, p=2, dim=2)
        cam_lost_mask = cam_norm2 < 1e-4 # [bsn, m]
        others_feat = others_feat.reshape(bsn, n, others_feat.shape[-1]) # [bsn, n, 7+1]
        dis = others_feat[:,:,-1] # [bsn, n]
        dis_lost_mask = dis < 1e-4 # [bsn, n]
        if (cam_lost_mask.all() or dis_lost_mask.any()):  # all cams are lost or any distance is lost
            print('all cams are lost or any distance is lost')
            prob = torch.zeros(others_prior_dir.shape[0], n, m).to(self.device) # [bsn, n, m]
            pos = others_feat[:,:,:3] # [bsn, n, 3]
            cov = torch.ones(others_prior_dir.shape[0], n, 1).to(self.device) * self.max_cov # [bsn, n, 1]
            scores = -torch.inf * torch.ones(others_prior_dir.shape[0], n+1, m+1).to(self.device) # [bsn, n+1, m+1]
            indices = -torch.ones(others_prior_dir.shape[0], n, 1).to(self.device).to(torch.int64) # [bsn, n, 1]       
        else:
            match_cam, prob, match_cos_similarity, var, indices, cov = self.cos_match(others_prior_dir, others_cam, cam_lost_mask)    
            pos = dis.unsqueeze(2) * match_cam # [bsn, n, 3] 
            scores = -torch.inf * torch.ones(others_prior_dir.shape[0], n+1, m+1).to(self.device) # [bsn, n+1, m+1]

        valid = indices > -1
        invalid_cov = torch.tensor(self.max_cov, dtype=cov.dtype, device=cov.device)
        cov = torch.where(valid, cov, invalid_cov)
        pos = torch.where(valid, pos, others_prior_pos)

        return prob, pos, cov, scores, indices

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
        # out_match = { 'cam': match_cam, 'prob': prob, 'cos_similarity': match_cos_similarity, 'var': var, 'indices': indices, 'cov': cov }

        return match_cam, prob, match_cos_similarity, var, indices, cov

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