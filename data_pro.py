from torch.utils.data import Dataset
import dgl
import torch

class ComPactedCSVDataset(dgl.data.CSVDataset):
    def __init__(self, swarm_num, *args, **kwargs):
        super(ComPactedCSVDataset, self).__init__(*args, **kwargs)
        batchgraph_list = []
        batch_timestamp = torch.zeros(int(len(self.graphs)/swarm_num), 2*swarm_num, dtype=torch.int64)
        batch_ref_id = torch.zeros(int(len(self.graphs)/swarm_num), 1*swarm_num, dtype=torch.int8)
        batch_world_pose_delta = torch.zeros(int(len(self.graphs)/swarm_num), 7*swarm_num, dtype=torch.float32)
        for i in range(0, len(self.graphs), swarm_num):
            batchgraph_list.append(dgl.batch(self.graphs[i:i + swarm_num]))
            timestamp = torch.zeros(1, swarm_num*2, dtype=torch.int64)
            ref_id = torch.zeros(1, swarm_num, dtype=torch.int8)
            world_pose_delta = torch.zeros(1, swarm_num*7, dtype=torch.float32)
            for j in range(swarm_num):
                ref_id[0, j] = self.data['ref_id'][i+j]
                timestamp[0, 2*j:2*j+2] = self.data['timestamp'][i+j]
                world_pose_delta[0, 7*j:7*j+7] = self.data['world_pose_delta'][i+j]
            batch_timestamp[int(i/swarm_num)] = timestamp
            batch_ref_id[int(i/swarm_num)] = ref_id
            batch_world_pose_delta[int(i/swarm_num)] = world_pose_delta

        self.graphs = batchgraph_list
        self.data['timestamp'] = batch_timestamp
        self.data['ref_id'] = batch_ref_id
        self.data['world_pose_delta'] = batch_world_pose_delta
    
    def process(self):
        super(ComPactedCSVDataset, self).process()

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
 
    def __len__(self):
        return len(self.sequences)
 
    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return sequence, label
    
def create_continues_sequences(dataset, tw, timestamp_thres):
    inout_seq = []
    L = len(dataset)
    for i in range(L - tw):
        # validate if the data inside a seq is consecutive!!!
        seq = dataset[i:i + tw] # tuple(graph_list, msg_dict)
        graph_list = seq[0]
        msg_dict = seq[1]
        last_t = None
        is_consecutive = True
        for row in range(msg_dict['timestamp'].shape[0]):
            # check timestamp
            t_decimal = float('0.' + str(msg_dict['timestamp'][row,1].item())[1:])
            t = msg_dict['timestamp'][row,0].item() + t_decimal
            if last_t is None:
                last_t = t
                continue
            if abs(t-last_t) > timestamp_thres:
                is_consecutive = False
                break
            last_t = t

        if not is_consecutive:
            continue
        
        inout_seq.append(seq)

    return inout_seq