import torch
import dgl.data
from dgl.dataloading import GraphDataLoader
from model.match_net_loading_single import GAT_TRANSFORMER
import loss_func as loss_func
import data_pro as dp
from args import get_args

def save_script(model, batched_graph, path, device):
    batched_graph = batched_graph.to(device)
    others_feat, cam_feat = batched_graph.ndata['feat']['others'], batched_graph.ndata['feat']['cam']
    cpu_device = torch.device("cpu")
    others_feat, cam_feat = others_feat.to(cpu_device), cam_feat.to(cpu_device)
    model = model.to(cpu_device)
    
    my_script = torch.jit.script(model)
    my_script.save(path)
    # my_trace = torch.jit.trace(model, (others_feat, cam_feat))
    # my_trace.save(path)
    

if __name__ == '__main__':
    args = get_args()
    model_file = args.model_file
    # val_dataset = dp.ComPactedCSVDataset(args.swarm_num, args.val_dataset)
    val_dataset = dgl.data.CSVDataset(args.val_dataset)
    print("Val dataset: ", len(val_dataset))
    assert args.batch_size == 1    
    val_dataloader = GraphDataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, shuffle=args.shuffle, pin_memory=torch.cuda.is_available())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    gat_transformer = GAT_TRANSFORMER(args, device).to(device)
    model = gat_transformer    
    ########### we don't need to optimize the model actually ############
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)
    
    pretrained_model = model_file + "/v1.pt"
    save_model_file = model_file + "/v1_scripted.pt"
    # Load pretrained model to continue training
    model.load_state_dict(torch.load(pretrained_model))

    model.eval()
    with torch.no_grad():
        for batched_graph, batched_msgs in val_dataloader:
            save_script(model, batched_graph, save_model_file, device)
            break  # Save only the first batch








