# pandaset_infos_train.pkl
# pandaset_dbinfos_train.pkl

import pickle
import gzip
import numpy as np

# 1. read uncertainty prediciton
file_path = 'output/uncertainty_dump/un_v4.pkl'
with open(file_path, 'rb') as f:
    uncertainty_map = pickle.load(f)
    #print(uncertainty_map)

device = 0

# 2. change pandaset_infos_train.pkl
file_path = 'pandaset/pandaset_infos_train.pkl'  # used to be 'pandaset/pandaset_infos_train_ori.pkl'
with open(file_path, 'rb') as f:
    pandaset_infos = pickle.load(f)
    #print(pandaset_infos)
for info in pandaset_infos:
    # import pdb;pdb.set_trace()
    #print(len(info['uncertainty']))
    with gzip.open(info['cuboids_path'], 'rb') as f:
        cuboids = pickle.load(f)
        cuboids = cuboids[cuboids["cuboids.sensor_id"] != 1 - device]
        #print(cuboids)
    seq_id = info['sequence']
    frame_id = str(info['frame_idx'])
    index_list = cuboids.index
    #print(index_list)
    names = cuboids['label']
    # import pdb;pdb.set_trace()
    uncertainty_list = []
    for i,idx in enumerate(index_list):
        name = names[idx]
        if name!='Car':
            uncertainty = np.array([-1 for i in range(7)])
        else:
            key = seq_id + '_' + frame_id + '_' + str(i)  # seq_id +
            #try:
            uncertainty = uncertainty_map[key]
            #except KeyError:
            #    uncertainty = np.array([-1 for i in range(7)])
        uncertainty_list.append(uncertainty)
    # import pdb;pdb.set_trace()
    #print(info)
    info['uncertainty'] = np.array(uncertainty_list)

file_path = 'pandaset/pandaset_infos_train_wconf_v4.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(pandaset_infos, f)


# 3. change pandaset_dbinfos_train.pkl
file_path = 'pandaset/pandaset_dbinfos_train.pkl'  # used to be _ori
with open(file_path, 'rb') as f:
    db_infos = pickle.load(f)

for info in db_infos['Car']:
    #print(info)
    seq_id = info['path'].split("/")[1].split("_")[0]
    frame_id = info['path'].split("/")[1].split("_")[1]
    gt_idx = info['gt_idx']
    key = seq_id + '_' + frame_id + '_' + str(gt_idx)
    uncertainty = uncertainty_map[key]
    info['uncertainty'] = uncertainty


file_path = 'pandaset/pandaset_dbinfos_train_wconf_v4.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(db_infos, f)
