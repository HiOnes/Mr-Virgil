import torch
import torch.nn.functional as F
from torch import nn, Tensor
from copy import deepcopy
from typing import Optional, List, Tuple

torch.set_printoptions(precision=4, sci_mode=False)

def MLP(channels: list, do_bn=True)-> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: Tensor, source: Tensor) -> Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for i, layer in enumerate(self.layers):
            name = self.names[i]
            if name == 'self':
                src0, src1 = desc0, desc1
            elif name == 'cross':
                src0, src1 = desc1, desc0
            else:
                raise ValueError(f'Unknown layer name: {name}')
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1

def log_sinkhorn_iterations(Z: Tensor, log_mu: Tensor, log_nu: Tensor, iters: int) -> Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: Tensor, alpha: Tensor, iters: int) -> Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = torch.tensor(1, dtype=scores.dtype, device=scores.device)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1

class KeypointEncoder(nn.Module):
    """ Joint encoding of prior or cam direction using MLPs"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.encoder = MLP([3, 32, 64, 64] + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts: Tensor) -> Tensor:
        return self.encoder(kpts.transpose(1, 2))

class GAT_TRANSFORMER(nn.Module):
    def __init__(self, args, device):
        
        super().__init__()        
        self.pos_pred = nn.Sequential(
            nn.Linear(3*2+2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.cov_pred = nn.Sequential(
            nn.Linear(3*2+2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self._reset_parameters()

        self.gnn = AttentionalGNN(feature_dim=args.embed_size, layer_names=['self']*4) # only self edges
        # self.gnn = AttentionalGNN(feature_dim=args.embed_size, layer_names=['self', 'cross']*4) # self and cross edges

        self.kenc = KeypointEncoder(args.embed_size)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.dot_attn = Dot_Attn(device).to(device)
        self.others_num = args.swarm_num - 1
        self.max_cam_num = args.max_cam_num
        self.dim = args.embed_size
        self.device = device
        self.max_cov = 10.0

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sinkhorn_match(self, prior_embed, cam_embed, cam_lost_mask, cam_dir):
        '''
        input:
            prior_embed: torch.Tensor, [bs, n, dim]
            cam_embed: torch.Tensor, [bs, m, dim]
            cam_lost_mask: torch.Tensor, [bs, m]
            cam_dir: torch.Tensor, [bs, m, 3]
        output:
            out_match: dict
        '''
        n, m = prior_embed.shape[1], cam_embed.shape[1]
        scores, prob = self.dot_attn(prior_embed, cam_embed, key_padding_mask=cam_lost_mask) # [bs, n, m]
        scores = scores / prior_embed.shape[-1] ** .5

         # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iters=100) # [bs, n+1, m+1]

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices # [bs, n], [bs, m]
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0) # [bs, n]
        zero_tensor = torch.tensor(0, dtype=scores.dtype, device=scores.device)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero_tensor) # [bs, n]
        valid0 = mutual0 & (mscores0 > 0.6) # [bsn, n]

        match_cam_index = indices0.unsqueeze(-1).repeat(1, 1, 3) # [bs, n, 3]
        match_cam = torch.gather(cam_dir, dim=1, index=match_cam_index) # [bs, n, 3]
        match_cos_similarity = mscores0.unsqueeze(-1) # [bs, n, 1]

        attn_dir = torch.bmm(prob, cam_dir) # [bs, n, 3]
        attn_dir = F.normalize(attn_dir, p=2.0, dim=2) # [bs, n, 3]
        var = torch.zeros(attn_dir.shape[0], n, 1).to(self.device) # [bs, n, 1]
        for i in range(n):
            mean = attn_dir[:, i, :].unsqueeze(1) # [bs, 1, 3]
            gap = cam_dir - mean # [bs, m, 3]
            gap_square = torch.norm(gap, p=2, dim=2, keepdim=True) # [bs, m, 1]
            bmm = torch.bmm(prob[:,i,:].unsqueeze(1), gap_square) # [bs, 1, 1]
            var[:, i, :] = bmm.squeeze(1) 

        invalid_index = torch.tensor(-1, dtype=indices0.dtype, device=indices0.device)
        indices0 = torch.where(valid0, indices0, invalid_index).unsqueeze(-1) # [bsn, n, 1]
        # out_match = { 'cam': match_cam, 'prob': prob, 'cos_similarity': match_cos_similarity, 'var': var, 'indices': indices0, 'scores': scores }
        return match_cam, prob, match_cos_similarity, var, indices0, scores
        
    def pos_cov_pred(self, match_cam, var, cos_similarity, others_feat):
        others_d = others_feat[:,:,-1].unsqueeze(2) # [bs, n, 1]
        others_prior_pos = others_feat[:,:,:3] # [bs, n, 3]
        attn_pos = others_d * match_cam # [bsn, n, 3]
        pos_feat = torch.cat([others_prior_pos, attn_pos, var, cos_similarity], dim=2) # [bs, n, 3*2+2]
        cov = self.cov_pred(pos_feat) # [bs, n, 1]
        cov = torch.clamp(cov, 1e-4, self.max_cov) # [bs, n, 1]
        outputs_relative_pos = self.pos_pred(pos_feat) # [bs, n, 3]
        outputs_pos = others_prior_pos + outputs_relative_pos # [bs, n, 3]
        return outputs_pos, cov
    

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

        others_cam = others_cam.reshape(bsn, -1, others_cam.shape[1]) # [bsn, m, 3]
        cam_norm2 = torch.norm(others_cam, p=2, dim=2)
        cam_lost_mask = cam_norm2 < 1e-4 # [bsn, m]

        others_encoder = self.kenc(others_prior_dir)
        cam_encoder = self.kenc(others_cam)
        others_gnn_feat, cam_gnn_feat = self.gnn(others_encoder, cam_encoder) # others_gnn_feat [bsn, dim, n], cam_gnn_feat [bsn, dim, m]
        others_gnn_feat, cam_gnn_feat = others_gnn_feat.transpose(1,2), cam_gnn_feat.transpose(1,2) # [bsn, n, dim], [bsn, m, dim]
        others_feat = others_feat.reshape(bsn, n, others_feat.shape[-1]) # [bsn, n, 7+1]
        dis_lost_mask = others_feat[:,:,-1] < 1e-4 # [bsn, n]
        if (cam_lost_mask.all() or dis_lost_mask.any()): # all cams are lost or any distances are lost
            prob = torch.zeros(others_prior_dir.shape[0], n, m).to(self.device) # [bsn, n, m]
            pos = others_feat[:,:,:3] # [bsn, n, 3]
            cov = torch.ones(others_prior_dir.shape[0], n, 1).to(self.device) * self.max_cov # [bsn, n, 1]
            scores = -torch.inf * torch.ones(others_prior_dir.shape[0], n+1, m+1).to(self.device) # [bsn, n+1, m+1]
            indices = -torch.ones(others_prior_dir.shape[0], n, 1).to(self.device).to(torch.int64) # [bsn, n, 1]
        else:
            match_cam, prob, match_cos_similarity, var, indices, scores = self.sinkhorn_match(others_gnn_feat, cam_gnn_feat, cam_lost_mask, others_cam)
            pos, cov = self.pos_cov_pred(match_cam, var, match_cos_similarity, others_feat)


        # if not valid, modify cov and pos
        valid = indices > -1
        invalid_cov = torch.tensor(self.max_cov, dtype=cov.dtype, device=cov.device)
        cov = torch.where(valid, cov, invalid_cov)
        pos = torch.where(valid, pos, others_prior_pos)

        # outputs = {'prob': out_prob, 'pos': out_pos, 'cov': out_cov, 'scores': out_scores, 'indices': out_indices}
        return prob, pos, cov, scores, indices

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