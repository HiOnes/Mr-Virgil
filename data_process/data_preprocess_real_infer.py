import numpy as np
import os
import pandas as pd
import random
import transforms3d as tfs
import math

train_dataset = ['real_los1', 'real_los2', 'real_los3', 'real_los4', "real_nlos1", "real_nlos2", "real_nlos3", "real_nlos4"]
# val_dataset = ['real_los5', 'real_los6', 'real_nlos5', 'real_nlos6']
val_dataset = ['real_nlos6']

dataset = {'train': train_dataset, 'val': val_dataset}
mode = 'val'

swarm_num = 5
max_cam_num = 8

fake_cam_prob = 0.0

start_time_cut = 0.1
end_time_cut = 0.1

# # big noise
# noise_imu_acc = 0.1
# noise_imu_gyr = 0.01
# noise_uwb = 0.1
# noise_cam = 0.035 # 2 degree
# noise_pos = 0.2

# small noise
noise_imu_acc = 0.01
noise_imu_gyr = 0.001
noise_uwb = 0.01
noise_cam = 0.0035
noise_pos = 0.1
noise_rot = 1.0 / 180 * math.pi # 1 degree
noise_world_pos = 0.02
noise_world_rot = 1.0 / 180 * math.pi # 1 degree

sample_step = 0.1 #s
pos_freq = 200 #Hz
imu_freq = 100 #Hz
pos_gap_line = int(pos_freq * sample_step)
imu_gap_line = int(imu_freq * sample_step)

use_pca = False

scene_path = "real_5robots/" # should match the swarm_num
pre_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/"
gnn_path = "gnn/" + scene_path
origin_folder = 'raw/' + scene_path

types_list = ['pos', 'imu', 'uwb', 'cam', 'world_pos']
time_gap_thres = {'pos': 0.008, 'imu': 0.015, 'uwb': 0.1, 'cam': 10, 'world_pos': 0.008} # cam set to 10s for consecutive infer, 0.03 for non-consecutive
# time_gap_thres = {'pos': 0.008, 'imu': 0.015, 'uwb': 0.015, 'cam': 10, 'world_pos': 0.008} # cam set to 10s for consecutive infer, 0.03 for non-consecutive
# graph level
graph_ind = -1
g_graphid_list = []
g_timestamp_list = []
g_refid_list = []
g_world_pose_delta_list = []
# node level
n_others_graphid_list = []
n_others_nodeid_list = []
n_others_feat = [] # [x, y, z, dis]
n_others_label_pos = [] # [x, y, z]
n_others_label_match = [] # [node_ind]
n_cam_graphid_list = []
n_cam_nodeid_list = []
n_cam_feat = [] # direction [x, y, z] after normalized

# edge level
e_others2cam_graphid_list = []
e_others2cam_srcid_list = []
e_others2cam_dstid_list = []
e_cam2others_graphid_list = []
e_cam2others_srcid_list = []
e_cam2others_dstid_list = []

def get_transform_matrix_from_pose_line(pose_line):
    pos = np.array(pose_line[1:4], float).reshape(3, 1)
    quat = np.array([pose_line[7], pose_line[4], pose_line[5], pose_line[6]], float)
    rot_mat = tfs.quaternions.quat2mat(quat)
    T = np.concatenate((rot_mat, pos), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
    return T

def get_pose_delta(pose_line0, pose_line1):
    T0 = get_transform_matrix_from_pose_line(pose_line0)
    T1 = get_transform_matrix_from_pose_line(pose_line1)
    T_delta = np.dot(np.linalg.inv(T0), T1)
    pos_delta = T_delta[0:3, 3]
    quat_delta = tfs.quaternions.mat2quat(T_delta[0:3, 0:3])
    pose_delta = pos_delta.tolist() + quat_delta.tolist() # [x, y, z, w, x, y, z]
    return pose_delta

class Cam_Object:
    def __init__(self, id, feat):
        self.id = id
        self.feat = feat

def read_file(data_name):
    print("-------------Read File: ", data_name, " ------------------")
    pos_map = {}
    imu_map = {}
    uwb_map = {}
    cam_map = {}
    world_pos_map = {}
    
    for i in range(swarm_num):
        pos_file = os.path.join(pre_path, origin_folder, data_name, "pos_" + str(i) + ".txt")
        imu_file = os.path.join(pre_path, origin_folder, data_name, "imu_" + str(i) + ".txt")
        uwb_file = os.path.join(pre_path, origin_folder, data_name, "uwb_" + str(i) + ".txt")
        cam_file = os.path.join(pre_path, origin_folder, data_name, "cam_" + str(i) + ".txt")
        world_pos_file = os.path.join(pre_path, origin_folder, data_name, "world_pos_" + str(i) + ".txt")
        pos_list = []
        imu_list = []
        uwb_list = []
        cam_list = []
        world_pos_list = []
        with open(pos_file, 'r') as f:
            for line in f.readlines():
                pos_list.append(line.strip('\n').rstrip().split(' '))
        start_time = float(pos_list[0][0]) + start_time_cut
        end_time = float(pos_list[-1][0]) - end_time_cut
        pos_list = [pos for pos in pos_list if float(pos[0]) >= start_time and float(pos[0]) <= end_time]
        pos_map[i] = pos_list

        with open(imu_file, 'r') as f:
            for line in f.readlines():
                imu_list.append(line.strip('\n').rstrip().split(' '))
        imu_list = [imu for imu in imu_list if float(imu[0]) >= start_time and float(imu[0]) <= end_time]
        imu_map[i] = imu_list

        with open(uwb_file, 'r') as f:
            for line in f.readlines():
                uwb_list.append(line.strip('\n').rstrip().split(' '))
        uwb_list = [uwb for uwb in uwb_list if float(uwb[0]) >= start_time and float(uwb[0]) <= end_time]
        uwb_map[i] = uwb_list

        with open(cam_file, 'r') as f:
            for line in f.readlines():
                cam_list.append(line.strip('\n').rstrip().split(' '))
        cam_list = [cam for cam in cam_list if float(cam[0]) >= start_time and float(cam[0]) <= end_time]
        cam_map[i] = cam_list

        with open(world_pos_file, 'r') as f:
            for line in f.readlines():
                world_pos_list.append(line.strip('\n').rstrip().split(' '))
        world_pos_list = [world_pos for world_pos in world_pos_list if float(world_pos[0]) >= start_time and float(world_pos[0]) <= end_time]
        for world_pos_line in world_pos_list:
            world_pose = [float(p) for p in world_pos_line[1:8]]
            world_pose_noise = add_noise(world_pose, 'world_pos')
            world_pos_line[1:8] = [str(p) for p in world_pose_noise]
        world_pos_map[i] = world_pos_list

    out = {'pos': pos_map, 'imu': imu_map, 'uwb': uwb_map, 'cam': cam_map, 'world_pos': world_pos_map}
    return out

def align_timestamp(line_list, line_index_, t_now):
    time_piece = 1e-6
    for line_index in range(line_index_, len(line_list)):
        line = line_list[line_index]
        if (t_now + time_piece <= float(line[0])):
            last_line_index = line_index - 1
            if last_line_index < 0:
                return None, line_index_
            return line_list[last_line_index], last_line_index # return last line
    return None, line_index_

def add_noise(data, type):
    if type == 'imu':
        data[0] += random.gauss(0, noise_imu_gyr)
        data[1] += random.gauss(0, noise_imu_gyr)
        data[2] += random.gauss(0, noise_imu_gyr)
        data[3] += random.gauss(0, noise_imu_acc)
        data[4] += random.gauss(0, noise_imu_acc)
        data[5] += random.gauss(0, noise_imu_acc)
    elif type == 'uwb':
        data += random.gauss(0, noise_uwb)
    elif type == 'cam':
        data[0] += random.gauss(0, noise_cam)
        data[1] += random.gauss(0, noise_cam)
        data[2] += random.gauss(0, noise_cam)
        # normalize
        norm = math.sqrt(data[0]**2 + data[1]**2 + data[2]**2)
        data[0] /= norm
        data[1] /= norm
        data[2] /= norm
    elif type == 'pos':
        data[0] += random.gauss(0, noise_pos)
        data[1] += random.gauss(0, noise_pos)
        data[2] += random.gauss(0, noise_pos)
        # to do orientation noise
        ori_quat = np.array([data[6], data[3], data[4], data[5]]) # w x y z
        roll, pitch, yaw = tfs.euler.quat2euler(ori_quat, axes='sxyz')
        roll += random.gauss(0, noise_rot)
        pitch += random.gauss(0, noise_rot)
        yaw += random.gauss(0, noise_rot)
        quat = tfs.euler.euler2quat(roll, pitch, yaw, axes='sxyz')
        data[3] = quat[1]
        data[4] = quat[2]
        data[5] = quat[3]
        data[6] = quat[0]

    elif type == 'world_pos':
        data[0] += random.gauss(0, noise_world_pos)
        data[1] += random.gauss(0, noise_world_pos)
        data[2] += random.gauss(0, noise_world_pos)
        # to do orientation noise
        ori_quat = np.array([data[6], data[3], data[4], data[5]]) # w x y z
        roll, pitch, yaw = tfs.euler.quat2euler(ori_quat, axes='sxyz')
        roll += random.gauss(0, noise_world_rot)
        pitch += random.gauss(0, noise_world_rot)
        yaw += random.gauss(0, noise_world_rot)
        quat = tfs.euler.euler2quat(roll, pitch, yaw, axes='sxyz')
        data[3] = quat[1]
        data[4] = quat[2]
        data[5] = quat[3]
        data[6] = quat[0]
    return data

def validate_data(list_k, line_index_k, t_now):
    line_k = dict.fromkeys(types_list, None)
    for types in types_list:
        if types == 'imu':
            continue
        line_k[types], line_index_k[types] = align_timestamp(list_k[types], line_index_k[types], t_now)

    for types in line_k.keys():
        if types == 'imu':
            continue
        if line_k[types] is None:
            return False
        if abs(t_now-float(line_k[types][0])) > time_gap_thres[types]:
            return False

    # extract last pos_line_ and imu_line_
    pos_line_index_ = line_index_k['pos'] - pos_gap_line
    if pos_line_index_ < 0:
        return False
    pos_line_ = list_k['pos'][pos_line_index_]
    # imu_line_index_ = imu_line_index + 1 - imu_gap_line
    # if imu_line_index_ < 0:
    #     return False

    # check pos_line complete
    if len(line_k['pos']) != 1+8*(swarm_num-1) or len(pos_line_) != 1+8*(swarm_num-1):
        print("pos_line not complete: ", len(line_k['pos']), len(pos_line_))
        return False
    
    # # check uwb_line complete
    # if len(line_k['uwb']) != 1+2*(swarm_num-1):
    #     print("uwb_line not complete: ", len(line_k['uwb']), ", only ", int((len(line_k['uwb'])-1)/2), " uwb")
    #     return False
    
    # # check cam_line not empty
    # if len(line_k['cam']) <= 1:
    #     print("cam_line not complete: ", len(line_k['cam']))
    #     return False

    return True


def process(graph_ind, k, data_list, line_index, t_now):
    pos_line, line_index['pos'] = align_timestamp(data_list['pos'], line_index['pos'], t_now)
    # imu_line, imu_line_index = align_timestamp(imu_list, imu_line_index, t_now)
    uwb_line, line_index['uwb'] = align_timestamp(data_list['uwb'], line_index['uwb'], t_now)
    cam_line, line_index['cam'] = align_timestamp(data_list['cam'], line_index['cam'], t_now)
    world_pos_line, line_index['world_pos'] = align_timestamp(data_list['world_pos'], line_index['world_pos'], t_now)

    # extract last pos_line_ and imu_line_
    pos_line_index_ = line_index['pos'] - pos_gap_line
    pos_line_ = data_list['pos'][pos_line_index_]
    # imu_line_index_ = imu_line_index + 1 - imu_gap_line
    world_pos_line_index_ = line_index['world_pos'] - pos_gap_line
    world_pos_line_ = data_list['world_pos'][world_pos_line_index_]
        
    # graph 
    graph_ind += 1
    g_graphid_list.append(graph_ind)
    integer_t, decimal_t = str(t_now).split('.')
    decimal_t = '1' + decimal_t # int can't start with 0
    timestamp = integer_t + ',' + decimal_t
    g_timestamp_list.append(timestamp)
    ref_id = str(k) + ','
    g_refid_list.append(ref_id)
    world_pose_delta = get_pose_delta(world_pos_line_, world_pos_line)
    world_pose_delta = ','.join(str(p) for p in world_pose_delta)
    g_world_pose_delta_list.append(world_pose_delta)

    # nodes_cam
    cam_consecutive = True
    if abs(t_now-float(cam_line[0])) > 0.03:
        cam_consecutive = False
    cam_obj_list = []
    if cam_consecutive:
        for i in range(1, len(cam_line), 4):
            cam_id = int(cam_line[i])
            cam_dir = [float(p) for p in cam_line[i+1:i+4]]
            cam_dir = add_noise(cam_dir, 'cam')
            cam_obj_list.append(Cam_Object(cam_id, cam_dir))
        if len(cam_obj_list) == 0:
            print("cam not found")
    else:
        print("cam not consecutive")
        for i in range(max_cam_num):
            cam_obj_list.append(Cam_Object(-1, [0, 0, 0]))
    
    # add fake cam
    for i in range(len(cam_obj_list), max_cam_num):
        if random.random() < fake_cam_prob:
            cam_dir = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
            # normalize cam dir
            norm = math.sqrt(cam_dir[0]**2 + cam_dir[1]**2 + cam_dir[2]**2)
            cam_dir = [cam_dir[0]/norm, cam_dir[1]/norm, cam_dir[2]/norm]
            cam_obj_list.append(Cam_Object(-1, cam_dir))

    random.shuffle(cam_obj_list)
    for i in range(len(cam_obj_list), max_cam_num):
        cam_obj_list.append(Cam_Object(-1, [0, 0, 0]))
    for i in range(len(cam_obj_list)):
        cam = cam_obj_list[i]
        cam_feat = ','.join(str(p) for p in cam.feat)
        n_cam_graphid_list.append(graph_ind)
        n_cam_nodeid_list.append(i)
        n_cam_feat.append(cam_feat)

    # nodes_others
    nodes_others_ind = -1
    for pos_index in range(1, len(pos_line_), 8):
        nodes_others_ind += 1
        id = int(pos_line_[pos_index])
        prior_pos = [float(p) for p in pos_line_[pos_index+1:pos_index+8]]
        prior_pos = add_noise(prior_pos, 'pos')

        for gt_index in range(1, len(pos_line), 8):
            if int(pos_line[gt_index]) == id:
                gt_pos = ','.join(str(p) for p in pos_line[gt_index+1:gt_index+8])
                break
                
        dis = -1.0
        for i in range(1, len(uwb_line), 2):
            if int(uwb_line[i]) == id:
                dis = float(uwb_line[i+1])
                dis = round(add_noise(dis, 'uwb'), 3)
                break

        cam_ind = -1
        for i in range(len(cam_obj_list)):
            cam_id = cam_obj_list[i].id
            if cam_id == id:
                cam_ind = i
                break
        
        others_feat = prior_pos + [dis]
        others_feat = ','.join(str(p) for p in others_feat)
        label_cam_ind = str(cam_ind) + ','
        n_others_graphid_list.append(graph_ind)
        n_others_nodeid_list.append(nodes_others_ind)
        n_others_feat.append(others_feat)
        n_others_label_pos.append(gt_pos)
        n_others_label_match.append(label_cam_ind)

        # edges
        for i in range(len(cam_obj_list)):
            e_others2cam_graphid_list.append(graph_ind)
            e_others2cam_srcid_list.append(nodes_others_ind)
            e_others2cam_dstid_list.append(i)
            e_cam2others_graphid_list.append(graph_ind)
            e_cam2others_srcid_list.append(i)
            e_cam2others_dstid_list.append(nodes_others_ind)

    return graph_ind, line_index


def generate_gnn_data(data):
    # print("-------------Generate GNN Data------------------")
    global graph_ind
    ref_id = 0
    line_index_map = {types: {k: 0 for k in range(swarm_num)} for types in types_list}

    for pos_line_index in range(0, len(data['pos'][ref_id]), pos_gap_line):
        ref_pos_line = data['pos'][ref_id][pos_line_index]
        ref_t = float(ref_pos_line[0])
        full_flag = True
        for k in range(swarm_num):
            list_k = {key : data[key][k] for key in data.keys()}
            line_index_k = {key : line_index_map[key][k] for key in line_index_map.keys()}
            validate_flag = validate_data(list_k, line_index_k, ref_t)
            if not validate_flag:
                full_flag = False
                break

        if not full_flag:
            continue

        for k in range(swarm_num):
            list_k = {key : data[key][k] for key in data.keys()}
            line_index_k = {key : line_index_map[key][k] for key in line_index_map.keys()}
            graph_ind, line_index_update_k = process(graph_ind, k, list_k, line_index_k, ref_t)
            for types in line_index_map.keys():
                line_index_map[types][k] = line_index_update_k[types]

                

if __name__ == '__main__':
    for data_name in dataset[mode]:
        data_map = read_file(data_name)
        generate_gnn_data(data_map)
    
    wrt_folder = os.path.join(pre_path, gnn_path, mode)
    if not os.path.exists(wrt_folder):
        os.makedirs(wrt_folder)

    graphs_file = os.path.join(wrt_folder, "graphs.csv")
    nodes_cam_file = os.path.join(wrt_folder, "nodes_cam.csv")
    nodes_others_file = os.path.join(wrt_folder, "nodes_others.csv")
    edges_cam2others_file = os.path.join(wrt_folder, "edges_cam2others.csv")
    edges_others2cam_file = os.path.join(wrt_folder, "edges_others2cam.csv")

    dataframe_graphs = pd.DataFrame({'graph_id': g_graphid_list, 'timestamp': g_timestamp_list, 'ref_id': g_refid_list, 'world_pose_delta': g_world_pose_delta_list})
    dataframe_graphs.to_csv(graphs_file, index=False, sep=',')
    dataframe_nodes_others = pd.DataFrame({'graph_id': n_others_graphid_list, 'node_id': n_others_nodeid_list, 'feat': n_others_feat, 'label_pos': n_others_label_pos, 'label_match': n_others_label_match})
    dataframe_nodes_others.to_csv(nodes_others_file, index=False, sep=',')
    dataframe_nodes_cam = pd.DataFrame({'graph_id': n_cam_graphid_list, 'node_id': n_cam_nodeid_list, 'feat': n_cam_feat})
    dataframe_nodes_cam.to_csv(nodes_cam_file, index=False, sep=',')
    dataframe_edges_cam2others = pd.DataFrame({'graph_id': e_cam2others_graphid_list, 'src_id': e_cam2others_srcid_list, 'dst_id': e_cam2others_dstid_list})
    dataframe_edges_cam2others.to_csv(edges_cam2others_file, index=False, sep=',')
    dataframe_edges_others2cam = pd.DataFrame({'graph_id': e_others2cam_graphid_list, 'src_id': e_others2cam_srcid_list, 'dst_id': e_others2cam_dstid_list})
    dataframe_edges_others2cam.to_csv(edges_others2cam_file, index=False, sep=',')
    print("-------------Generate GNN Data Done------------------")