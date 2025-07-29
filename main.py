# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # set gpu id
import torch
from dgl.dataloading import GraphDataLoader
from model.match_net import GAT_TRANSFORMER
import loss_func as loss_func
import utils
import data_pro as dp
import pgo_theseus as pgo
import math
from args import get_args

def train_loops(model, dataloader, set_loss, args, device, opt):
    loss_avg = {"total": 0, "match": 0, "pos": 0, "rot": 0, "precision": 0, "cov": 0, "recall": 0}
    batch_cnt = 0
    model.train()
    for batched_graph, batched_msgs in dataloader:
        loss, _ = train_val_step(model, batched_graph, batched_msgs, set_loss, args, device)
        opt.zero_grad()
        loss['total'].backward()
        opt.step()
        batch_cnt += 1
        for k, v in loss.items():
            loss_avg[k] += v.detach().item()

    assert batch_cnt > 0, "Batch count should be greater than 0"
    for k in loss_avg.keys():
        loss_avg[k] /= batch_cnt
    loss_avg['pos'] = math.sqrt(loss_avg['pos']) # RMSE

    return loss_avg

def val_loops(model, dataloader, set_loss, args, device):
    loss_avg = {"total": 0, "match": 0, "pos": 0, "rot": 0, "precision": 0, "cov": 0, "recall": 0}
    batch_cnt = 0
    model.eval()
    with torch.no_grad():
        last_preds, last_t = None, None
        for batched_graph, batched_msgs in dataloader:
            if args.mode == 'infer':
                last_t = process_batchgraph(batched_graph, batched_msgs, last_t, last_preds, batch_cnt, args)
            loss, out = train_val_step(model, batched_graph, batched_msgs, set_loss, args, device)
            if args.mode == 'infer':
                if args.use_pgo:
                    last_preds = out['pose'].reshape(-1, 7)
                else:
                    last_preds = out['pos'].reshape(-1, 3)
            
            batch_cnt += 1

            for k, v in loss.items():
                loss_avg[k] += v.detach().item()

        assert batch_cnt > 0, "Batch count should be greater than 0"
        for k in loss_avg.keys():
            loss_avg[k] /= batch_cnt
        loss_avg['pos'] = math.sqrt(loss_avg['pos']) # RMSE

    return loss_avg

def train_val_step(model, batched_graph, batched_msgs, set_loss, args, device):
    batched_graph = batched_graph.to(device)
    if args.mode == 'infer':
        assert args.batch_size == 1, "Batch size should be 1 for inference"
        update_prior_from_odom(batched_graph, batched_msgs, args)
    label_match = batched_graph.ndata['label_match']['others'] # [batch_size*swarm_num*(swarm_num-1), 1]
    label_pose = batched_graph.ndata['label_pos']['others'] # [batch_size*swarm_num*(swarm_num-1), 7]
    prior_pose = batched_graph.ndata['feat']['others'][:, :7] # [batch_size*swarm_num*(swarm_num-1), 7]
    mn_out = model(batched_graph.ndata['feat']['others'], batched_graph.ndata['feat']['cam']) # match net outputs
    mn_out_pos = mn_out['pos'] # [batch_size*swarm_num, (swarm_num-1), 3]
    mn_out_cov = mn_out['cov'] # [batch_size*swarm_num, (swarm_num-1), 1]
    pred_pose = None
    ############################# RUN PGO #############################
    if args.use_pgo:
        pgo_out_pose = pgo.run_pgo(mn_out_pos, mn_out_cov, prior_pose, swarm_num=args.swarm_num, device=device)
        pred_pose = pgo_out_pose
        loss_type = 'match_6dpose'
    else:
        loss_type = 'match_3dpos'
    out = {'cov': mn_out_cov, 'pos': mn_out_pos, 'pose': pred_pose, 'scores': mn_out['scores'], 'indices': mn_out['indices']}
    target = {'match': label_match, 'pose': label_pose}
    loss = set_loss(out, target, supervised_type=loss_type)

    return loss, out

def update_prior_from_odom(batched_graph, batched_msgs, args):
    prior = batched_graph.ndata['feat']['others'][:, :-1] # [bs*(n+1)*n, 7] [x, y, z, qx, qy, qz, qw]
    prior = utils.xyz_xyzw_2_xyz_wxyz(prior).reshape(-1, args.swarm_num, args.swarm_num-1, 7)  # [bs, n+1, n, 7] [x, y, z, qw, qx, qy, qz]
    world_pose_delta = batched_msgs['world_pose_delta'].to(batched_graph.device)   # [bs, (n+1)*7] [x, y, z, qw, qx, qy, qz]
    world_pose_delta = world_pose_delta.reshape(-1, args.swarm_num, 7) # [bs, n+1, 7] 
    updated_prior = torch.zeros_like(prior) # [bs, n+1, n, 7] [x, y, z, qw, qx, qy, qz]
    for i in range(args.swarm_num):
        cnt = 0
        world_pose_delta_i = world_pose_delta[:, i, :] # [bs, 7]
        for j in range(args.swarm_num):
            if i == j:
                continue
            world_pose_delta_j = world_pose_delta[:, j, :] # [bs, 7]
            # [bs, 7] [x, y, z, qw, qx, qy, qz]
            updated_prior[:, i, cnt, :] = utils.batched_quatPose_update_torch(world_pose_delta_i, prior[:, i, cnt, :], world_pose_delta_j)
            cnt += 1
    updated_prior = utils.xyz_wxyz_2_xyz_xyzw(updated_prior.flatten(0,2)) # [bs*(n+1)*n, 7] [x, y, z, qx, qy, qz, qw]
    batched_graph.ndata['feat']['others'][:, :-1] = updated_prior

def process_batchgraph(batched_graph, batched_msgs, last_t, last_preds, val_cnt, args):
    is_consecutive = True
    # get timestamp
    t_decimal = float('0.' + str(batched_msgs['timestamp'][0,1].item())[1:])
    t = batched_msgs['timestamp'][0,0].item() + t_decimal
    if last_t is not None and abs(t-last_t) > args.timestamp_thres:
        print('Timestamp not consecutive -> last_t:', last_t, ', t:', t)
        is_consecutive = False
    last_t = t
    label_pose =batched_graph.ndata['label_pos']['others'] # [batch_size*swarm_num*(swarm_num-1), 7]
    if val_cnt > 0 and is_consecutive:
        batched_graph.ndata['feat']['others'][:, :last_preds.shape[-1]] = last_preds[:, :]
    else:
        batched_graph.ndata['feat']['others'][:, :7] = label_pose[:, :]
        print("reset prior")

    return last_t




if __name__ == '__main__':
    args = get_args()

    if args.mode in ['train', 'finetune']:
        train_dataset = dp.ComPactedCSVDataset(args.swarm_num, args.train_dataset)
        train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=args.shuffle, pin_memory=torch.cuda.is_available())
        print(f"Train dataset: {args.train_dataset}, Len: {len(train_dataset)}")

    val_dataset = dp.ComPactedCSVDataset(args.swarm_num, args.val_dataset)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, shuffle=args.shuffle, pin_memory=torch.cuda.is_available())
    print(f"Val dataset: {args.val_dataset}, Len: {len(val_dataset)}")
    print(f"Mode: {args.mode}, Batch size: {args.batch_size}, Learning rate: {args.lr}, Shuffle: {args.shuffle}, Model file: {args.model_file}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT_TRANSFORMER(args, device).to(device)

    if args.mode == 'finetune':
        # Freeze Match Net (GNN, K-encoder, Bin_score) when finetune
        for name, param in model.named_parameters():
            if "gnn" in name or "kenc" in name or "bin_score" in name:
                param.requires_grad = False

    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)
    set_loss = loss_func.SetLoss()
    
    best_model_file = args.model_file + "/best_model.pt"
    pretrained_model = args.model_file + "/v0.pt"
    best_val_loss = 1e4
    # Load pretrained model to continue training
    model.load_state_dict(torch.load(pretrained_model))

    for epoch in range(args.epochs):

        ##### train and validation loops #####
        if args.mode == 'train' or args.mode == 'finetune':
            train_loss = train_loops(model, train_dataloader, set_loss, args, device, opt)
        val_loss = val_loops(model, val_dataloader, set_loss, args, device)
        
        ###### print training and validation loss && save checkpoints ######
        if args.mode in ['train', 'finetune']:
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Train-- Loss {train_loss['total']:.4f} | Match {train_loss['match']:.4f} | Precision {train_loss['precision']:.4f} | "
                    f"Recall {train_loss['recall']:.4f} | Pos {train_loss['pos']:.4f} | Rot {train_loss['rot']:.4f} | Cov {train_loss['cov']:.4f} || "
                    f"Validation-- Loss {val_loss['total']:.4f} | Match {val_loss['match']:.4f} | Precision {val_loss['precision']:.4f} | "
                    f"Recall {val_loss['recall']:.4f} | Pos {val_loss['pos']:.4f} | Rot {val_loss['rot']:.4f} | Cov {val_loss['cov']:.4f}")
                
            if val_loss['total'] < best_val_loss:
                print(f"***Epoch {epoch} | Train-- Loss {train_loss['total']:.4f} | Match {train_loss['match']:.4f} | Precision {train_loss['precision']:.4f} | "
                    f"Recall {train_loss['recall']:.4f} | Pos {train_loss['pos']:.4f} | Rot {train_loss['rot']:.4f} | Cov {train_loss['cov']:.4f} || "
                    f"Validation-- Loss {val_loss['total']:.4f} | Match {val_loss['match']:.4f} | Precision {val_loss['precision']:.4f} | "
                    f"Recall {val_loss['recall']:.4f} | Pos {val_loss['pos']:.4f} | Rot {val_loss['rot']:.4f} | Cov {val_loss['cov']:.4f}***")
                best_val_loss = val_loss['total']
                torch.save(model.state_dict(), best_model_file)
                torch.save(model.state_dict(), args.model_file + "/e-{:04d}.pt".format(epoch))

        elif args.mode in ['eval', 'infer']:
            print(f"***Epoch {epoch} | Validation-- Loss {val_loss['total']:.4f} | Match {val_loss['match']:.4f} | Precision {val_loss['precision']:.4f} | "
              f"Recall {val_loss['recall']:.4f} | Pos {val_loss['pos']:.4f} | Rot {val_loss['rot']:.4f} | Cov {val_loss['cov']:.4f}***")
            break