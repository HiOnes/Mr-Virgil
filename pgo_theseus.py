import torch
import theseus as th
import utils

def generate_theseus_input(opt_pose, ref_id):
    '''
    input:
        opt_pose: torch.tensor, [bs, n, 7]
        ref_id: int
    output:
        inputs: dict
    '''
    opt_pose_clone = opt_pose.clone()
    inputs = {}
    for i in range(opt_pose_clone.size(1)):
        if i == ref_id:
            continue
        inputs[f"opt_pose_{i}"] = th.SE3(opt_pose_clone[:, i, :])
    return inputs

def cross_rel_pos_err_func(optim_vars, aux_vars):
    node_i, node_j = optim_vars # [bs, se3]
    [edge_ij] = aux_vars # [bs, se3]
    err = edge_ij.translation() - node_i.rotation().inverse().rotate(node_j.translation() - node_i.translation())   # [bs, 3]
    return err.tensor

def self_rel_pos_err_func(optim_vars, aux_vars):
    [node_i] = optim_vars # [bs, se3]
    [edge_ij] = aux_vars # [bs, se3]
    err = edge_ij.translation() - node_i.rotation().inverse().rotate(-node_i.translation())   # [bs, 3]
    return err.tensor

def build_pgo(ref_id, opt_pose, poses_all, pos_cov, device='cuda:0'):
    '''
    input:
        ref_id: int, reference node id
        opt_pose: torch.tensor, [bs, n, 7]
        poses_all: torch.tensor, [bs, n, n, 7]
        pos_cov: torch.tensor, [bs*n, n-1, 1]
        device: str, device id
    output:
        out_pose: torch.tensor, [bs, n, 7]
    '''
    bs, n = poses_all.size(0), poses_all.size(1)
    objective = th.Objective().to(device)
    poses = {}
    for i in range(n):
        if i == ref_id:
            continue
        poses[i] = th.SE3(opt_pose[:, i], name=f"opt_pose_{i}")

    ############## pos_weight ###############
    pos_cov = pos_cov.reshape(-1, n, n-1, 1) * 10.0 # [bs, n, n-1, 1]
    cov_inv = (1.0 / pos_cov).repeat(1, 1, 1, 3) # [bs, n, n-1, 3]
    rel_pos_weight = torch.sqrt(cov_inv + 1e-6) # [bs, n, n-1, 3]
    prior_pose_weight = rel_pos_weight.repeat(1, 1, 1, 2) * 1.0 # [bs, n, n-1, 6] # used to be 0.2

    ################# rel pos cost ###############
    for i in range(n):
        if i == ref_id:
            continue
        j_cnt = 0
        for j in range(n):
            if i == j:
                continue
            aux_vars = [th.SE3(poses_all[:, i, j], name=f"cross_edge_{i}_{j}")]
            w = th.DiagonalCostWeight(rel_pos_weight[:, i, j_cnt, :])
            j_cnt += 1
            if j == ref_id:
                ###### self rel pos cost ######
                optim_vars = [poses[i]]
                objective.add(th.AutoDiffCostFunction(optim_vars=optim_vars, err_fn=self_rel_pos_err_func, dim=3, aux_vars=aux_vars, name=f"self_rel_pos_cost_{i}_{j}", cost_weight=w))
            else:
                ###### cross rel pos cost ######
                optim_vars = poses[i], poses[j]
                objective.add(th.AutoDiffCostFunction(optim_vars=optim_vars, err_fn=cross_rel_pos_err_func, dim=3, aux_vars=aux_vars, name=f"cross_rel_pos_cost_{i}_{j}", cost_weight=w))

    ############# prior pose cost ############
    i_cnt = 0
    for i in range(n):
        if i == ref_id:
            continue
        w = th.DiagonalCostWeight(prior_pose_weight[:, ref_id, i_cnt, :])
        i_cnt += 1
        aux_targets = th.SE3(poses_all[:, ref_id, i], name=f"prior_node_{i}")
        objective.add(th.Difference(var=poses[i], target=aux_targets, cost_weight=w, name=f"prior_pose_cost_{i}"))      
    
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=5,
        step_size=1.0,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
        vectorize=True
    )

    theseus_optim = th.TheseusLayer(optimizer).to(device)
    theseus_inputs = generate_theseus_input(opt_pose, ref_id)
    # objective.update(theseus_inputs)
    solution, _ = theseus_optim.forward(theseus_inputs, optimizer_kwargs={"verbose": False})

    out_pose = opt_pose.clone()
    for i in range(n):
        if i == ref_id:
            continue
        out_pose[:, i, :] = th.SE3(tensor=solution[f"opt_pose_{i}"]).to_x_y_z_quaternion()

    return out_pose


def run_pgo(mn_out_pos, mn_out_cov, prior_pose, swarm_num=16, device='cuda:0'):
    '''
    input:
        mn_out_pos: torch.tensor, [batch_size*swarm_num, (swarm_num-1), 3]
        mn_out_cov: torch.tensor, [batch_size*swarm_num, (swarm_num-1), 1]
        prior_pose: torch.tensor, [batch_size*swarm_num*(swarm_num-1), 7]
        swarm_num: int
        device: str, device id
    output:
        pgo_out_pose: torch.tensor, [batch_size*swarm_num, (swarm_num-1), 7]
    '''
    bs = int(mn_out_pos.size(0) / swarm_num)
    ### update pose ###
    update_pose = prior_pose.reshape(-1, swarm_num-1, 7) # [batch_size*swarm, (swarm_num-1), 7]
    update_pose[:, :, :3] = mn_out_pos
    update_pose = update_pose.reshape(-1, swarm_num, swarm_num-1, 7) # [batch_size, swarm_num, (swarm_num-1), 7]

    ### padding ###
    poses_all = torch.zeros(bs, swarm_num, swarm_num, 7).to(device) # [batch_size, swarm_num, swarm_num, 7]
    empty_transform = torch.tensor([[0, 0, 0, 0, 0, 0, 1]]).repeat(bs, 1).to(device) # [batch_size, 7]
    for i in range(swarm_num):
        cnt = 0
        for j in range(swarm_num):
            if i == j:
                poses_all[:, i, j, :] = empty_transform
            else:
                poses_all[:, i, j, :] = update_pose[:, i, cnt, :]
                cnt += 1

    poses_all_xyzwxyz = utils.xyz_xyzw_2_xyz_wxyz(poses_all)

    opt_poses = torch.zeros_like(poses_all).to(device)
    for ref_id in range(swarm_num):
        param_pose_xyzwxyz = poses_all_xyzwxyz[:, ref_id, :, :] # [batch_size, swarm_num, 7]
        #### build and run pgo ####
        opt_pose_xyzwxyz = build_pgo(ref_id, param_pose_xyzwxyz, poses_all_xyzwxyz, mn_out_cov, device=device)
                
        opt_poses[:, ref_id, :, :] = opt_pose_xyzwxyz

    pgo_out_pose = torch.zeros(bs, swarm_num, swarm_num-1, 7).to(device) 
    for i in range(swarm_num):
        cnt = 0
        for j in range(swarm_num):
            if i == j:
                continue
            else:
                pgo_out_pose[:, i, cnt, :] = opt_poses[:, i, j, :]
                cnt += 1   

    pgo_out_pose = utils.xyz_wxyz_2_xyz_xyzw(pgo_out_pose)
    
    return pgo_out_pose
