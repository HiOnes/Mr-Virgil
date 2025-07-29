import torch.nn.functional as F
from torch import nn
import torch
import utils

class SetLoss(nn.Module):
    def __init__(self, w_match: float = 1.0, w_pos: float = 1.0, w_rot: float = 1.0, w_cov: float = 0.1):
        super().__init__()
        self.w_match = w_match
        self.w_pos = w_pos
        self.w_rot = w_rot
        self.w_cov = w_cov
    
    def forward(self, out, target, supervised_type):
        if supervised_type == 'match_6dpose':
            return self.forward_match_6dpose(out, target)
        elif supervised_type == 'match_3dpos':
            return self.forward_match_3dpos(out, target)
        else:
            raise ValueError("Invalid supervised_type")

    def forward_match_3dpos(self, out, target):
        '''
        input:
            out: dict
                scores: torch.tensor, [bs*(n+1), n+1, m+1] in log space negative
                indices: torch.tensor, [bs*(n+1), n, 1]
                cov: torch.tensor, [bs*(n+1), n, 1]
                pos: torch.tensor, [bs*(n+1), n, 3]
            target: dict
                pose: torch.tensor, [bs*(n+1)*n, 7]
                match: torch.tensor, [bs*(n+1)*n, 1] others2cam
        output:
            loss: dict
        '''

        ################### match loss ##################
        match_res = loss_match_func(target['match'], out['scores'], out['indices'])
        loss_match, recall, precision = match_res['cost'], match_res['recall'], match_res['precision']

        ################### pose loss ##################
        label_pose = target['pose'] # [bs*(n+1)*n, 7]
        out_pos = out['pos'].flatten(0,1) # [bs*(n+1)*n, 3]
        loss_pos = F.mse_loss(out_pos[:, :3], label_pose[:, :3]) * 3 # Positional Error
        loss_rot = torch.tensor(0.0).to(loss_pos.device)

        ################### cov ml loss ##################
        out_cov = out['cov'].flatten(0,1) # [bs*(n+1)*n, 1]
        loss_cov = loss_cov_func(label_pose, out_pos, out_cov, match_res['pred_valid_mask'])

        ################ final loss ################
        total_loss = loss_match*self.w_match + loss_pos*self.w_pos + loss_cov*self.w_cov

        loss = {'total': total_loss, 'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'precision': precision, 'cov': loss_cov, 'recall': recall}

        return loss
    
    def forward_match_6dpose(self, out, target):
        '''
        input:
            out: dict
                scores: torch.tensor, [bs*(n+1), n+1, m+1] in log space negative
                indices: torch.tensor, [bs*(n+1), n, 1]
                cov: torch.tensor, [bs*(n+1), n, 1]
                pos: torch.tensor, [bs*(n+1), n, 3] from front end
                pose: torch.tensor, [bs, n+1, n, 7] from back end
            target: dict
                pose: torch.tensor, [bs*(n+1)*n, 7]
                match: torch.tensor, [bs*(n+1)*n, 1] others2cam
        output:
            loss: dict
        '''

        ################### match loss ##################
        match_res = loss_match_func(target['match'], out['scores'], out['indices'])
        loss_match, recall, precision = match_res['cost'], match_res['recall'], match_res['precision']

        ################### pose loss of backend ##################
        label_pose = target['pose'] # [bs*(n+1)*n, 7]
        out_pose = out['pose'].reshape(-1, 7) # [bs*(n+1)*n, 7]
        loss_pos = F.mse_loss(out_pose[:, :3], label_pose[:, :3]) * 3 # Positional Error
        out_rot = utils.keep_w_positive(out_pose[:, 3:7])
        label_rot = utils.keep_w_positive(label_pose[:, 3:7])
        loss_rot = F.mse_loss(out_rot, label_rot)

        ################### cov ml loss of frontend ##################
        out_pos = out['pos'].flatten(0,1) # [bs*(n+1)*n, 3]
        out_cov = out['cov'].flatten(0,1) # [bs*(n+1)*n, 1]
        loss_cov = loss_cov_func(label_pose, out_pos, out_cov, match_res['pred_valid_mask'])

        ################ final loss ################
        total_loss = loss_match*self.w_match + loss_pos*self.w_pos + loss_rot*self.w_rot + loss_cov*self.w_cov
        loss = {'total': total_loss, 'match': loss_match, 'pos': loss_pos, 'rot': loss_rot, 'precision': precision, 'cov': loss_cov, 'recall': recall}

        return loss

def loss_match_func(label_indices, out_scores, out_indices, mask=None):
    '''
    input:
        label_indices: torch.tensor, [bs*swarm_num*n, 1]
        out_scores: torch.tensor, [bs*swarm_num, n+1, m+1]
        out_indices: torch.tensor, [bs*swarm_num, n, 1]
        mask: Optional, torch.tensor, [bs*swarm_num*n, 1]
    output:
        match_res: dict
    '''
    n, m = out_scores.size(1) - 1, out_scores.size(2) - 1
    li = label_indices.clone()
    label_valid_mask = (li != -1).flatten() # [bs*swarm_num*n]
    li[~label_valid_mask, :] = m
    scores = out_scores[:, :-1, :].flatten(0, 1) # [bs*swarm_num*n, m+1]
    cost = -torch.gather(scores, 1, li) # [bs*swarm_num*n, 1]

    ################# match recall & precision ################
    out_indices = out_indices.flatten(0,1) # [bs*swarm_num*n, 1]
    indices_success = out_indices == label_indices # [bs*swarm_num*n, 1]
    pred_valid_mask = (out_indices != -1).flatten() # [bs*swarm_num*n]

    if mask is not None:
        label_valid_mask = label_valid_mask & mask.flatten()
        pred_valid_mask = pred_valid_mask & mask.flatten()

    # calculate recall
    if label_valid_mask.any():
        valid_match_success = indices_success[label_valid_mask, :]
        recall = valid_match_success.float().mean()
    else:
        recall = torch.tensor(1.0).to(out_scores.device)
    # calculate precision
    if pred_valid_mask.any():
        valid_match_success = indices_success[pred_valid_mask, :]
        precision = valid_match_success.float().mean()
    else:
        precision = torch.tensor(1.0).to(out_scores.device)

    match_res = {'cost': cost.mean(), 'precision': precision, 'recall': recall,
                 'label_valid_mask': label_valid_mask, 'pred_valid_mask': pred_valid_mask}
    
    return match_res

def loss_cov_func(label, pred, cov, mask=None, dim=3):
    '''
    calculate the covariance loss of predicted [ dis(dim=1) or pos(dim=3) or pose(dim=6) ]
    input:
        label: torch.tensor, [bs, 1 or 3 or 7]
        pred: torch.tensor, [bs, 1 or 3 or 7]
        cov: torch.tensor, [bs, 1 or 3 or 6]
        mask: torch.tensor, [bs]
        dim: int, 1 or 3 or 6
    output:
        loss_cov: torch.tensor
    '''
    if mask is None:
        mask = torch.ones_like(label[:, 0]).bool() # [bs]
    if (~mask).all():
        return torch.tensor(5.0).to(label.device)
    assert min(label.shape[-1], pred.shape[-1]) >= dim

    if dim == 1:
        diff = (pred[mask, :] - label[mask, :]).unsqueeze(-1) # [valid_num, 1, 1]
    elif dim == 3:
        diff = (pred[mask, :3] - label[mask, :3]).unsqueeze(-1) # [valid_num, 3, 1]
    elif dim == 6:
        diff = (pred[mask, :3] - label[mask, :3]).unsqueeze(-1) # [valid_num, 3, 1]
        quat_diff = utils.batched_quat_diff_torch(pred[mask, 3:7], label[mask, 3:7], w_first=False) # [valid_num, 4]
        euler_diff = utils.batched_quat_to_euler_torch(quat_diff, w_first=True) # [valid_num, 3]
        diff = torch.cat((diff, euler_diff.unsqueeze(-1)), dim=1) # [valid_num, 6, 1]

    if cov.shape[1] == dim:
        cov = cov[mask, :] # [valid_num, dim]
    elif cov.shape[1] == 1:
        cov = cov[mask, :].repeat(1, dim) # [valid_num, dim]
    else:
        raise ValueError("out_cov shape is not valid")
    cov_mat = torch.diag_embed(cov) # [valid_num, dim, dim]
    info_mat = torch.inverse(cov_mat) # [valid_num, dim, dim]
    det_cov = torch.det(cov_mat).unsqueeze(-1) # [valid_num, 1]
    loss_out_cov = torch.log(det_cov)
    loss_weighted_diff = torch.matmul(torch.matmul(diff.transpose(1, 2), info_mat), diff).squeeze(-1) # [valid_num, 1]
    loss_cov = (loss_weighted_diff + loss_out_cov*0.04).mean()

    return loss_cov