import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=["train", "finetune", "eval", "infer"], default="eval")
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=1)   # 1 for inference
    parser.add_argument('--swarm_num', type=int, default=16)   # real: 5, sim: 4  8  12  16
    parser.add_argument('--max_cam_num', type=int, default=20) # real: 8, sim: 8  12 16  20
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_pgo', type=bool, default=False)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--model_file', type=str, default="./checkpoints/model_loading/sim", help="folder to load and save model checkpoints")
    parser.add_argument('--train_dataset', type=str, default="./data/gnn/sim_16robots/train0_world_pose_delta")
    parser.add_argument('--val_dataset', type=str, default="./data/gnn/sim_16robots/val0_world_pose_delta")
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--timestamp_thres', type=float, default=0.15)

    # Params setting automatically by the system
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args
