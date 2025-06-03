"""
Define pytorch dataloader for DeepAttnMISL
"""
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader# 修改适配院内10%...

def set_mil_seed(seed):   # 新增
    global GLOBAL_MIL_SEED
    GLOBAL_MIL_SEED = seed

def get_mil_seed():       # 新增
    global GLOBAL_MIL_SEED
    return GLOBAL_MIL_SEED if GLOBAL_MIL_SEED is not None else 666

def worker_init_fn(worker_id):  # 新增
    seed = get_mil_seed()
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

class MIL_dataloader():
    def __init__(self, data_path, cluster_num=10, train=True, seed=None):
        self.cluster_num = cluster_num
        self.seed = seed if seed is not None else get_mil_seed()
        
        if train:
            statuses = []
            for path in data_path:
                npz = np.load(path)
                status = npz['status']
                if np.ndim(status) > 0:
                    status = status[0]
                statuses.append(status)
            statuses = np.array(statuses)

            def stratified_split(data_paths, statuses, test_size=0.1, random_state=66, max_retry=100):
                for _ in range(max_retry):
                    train_paths, val_paths, train_st, val_st = train_test_split(
                        data_paths, statuses, test_size=test_size, random_state=random_state, shuffle=True, stratify=None)
                    if np.any(val_st == 1):
                        return train_paths, val_paths
                    random_state += 1
                raise RuntimeError("Failed to split data with event in validation set after {} retries.".format(max_retry))

            X_train, X_test = stratified_split(data_path, statuses, test_size=0.1, random_state=self.seed)

            traindataset = MIL_dataset(list_path=X_train, cluster_num=cluster_num, train=train,
                                       transform=transforms.Compose([ToTensor(cluster_num)]))
            traindataloader = DataLoader(
                traindataset, batch_size=1, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn
            )

            valdataset = MIL_dataset(list_path=X_test, train=False, cluster_num=cluster_num,
                                     transform=transforms.Compose([ToTensor(cluster_num)]))
            valdataloader = DataLoader(
                valdataset, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn
            )

            self.dataloader = [traindataloader, valdataloader]

        else:
            testdataset = MIL_dataset(list_path=data_path, cluster_num=cluster_num, train=False,
                                     transform=transforms.Compose([ToTensor(cluster_num)]))
            testloader = DataLoader(
                testdataset, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn
            )
            self.dataloader = testloader

    def get_loader(self):
        return self.dataloader

class MIL_dataset(Dataset):
    def __init__(self, list_path, cluster_num,  transform=None, train=True):
        self.list_path = list_path
        self.random = train
        self.transform = transform
        self.cluster_num = cluster_num

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):
        img_path = self.list_path[idx]
        try:
            Batch_set = []
            surv_time_train = []
            status_train = []
            all_vgg = []
            vgg_clus = [[] for i in range(self.cluster_num)]
            Train_vgg_file = np.load(img_path)
            cur_vgg = Train_vgg_file['vgg_features']
            cur_patient = Train_vgg_file['pid']
            cur_time = Train_vgg_file['time']
            cur_status = Train_vgg_file['status']
            cur_path = Train_vgg_file['img_path']
            cur_cluster = Train_vgg_file['cluster_num']

            for id, each_patch_cls in enumerate(cur_cluster):
                vgg_clus[each_patch_cls].append(cur_vgg[id])

            Batch_set.append((cur_vgg, cur_patient, cur_status, cur_time, cur_cluster))
            np_vgg_fea = []
            mask = np.ones(self.cluster_num, dtype=np.float32)
            for i in range(self.cluster_num):
                if len(vgg_clus[i]) == 0:
                    clus_feat = np.zeros((1, 4096), dtype=np.float32)
                    mask[i] = 0
                else:
                    if self.random:
                        curr_feat = vgg_clus[i]
                        ind = np.arange(len(curr_feat))
                        np.random.seed(get_mil_seed())
                        np.random.shuffle(ind)
                        clus_feat = np.asarray([curr_feat[i] for i in ind])
                    else:
                        clus_feat = np.asarray(vgg_clus[i])
                clus_feat = np.swapaxes(clus_feat, 1, 0)
                clus_feat = np.expand_dims(clus_feat, 1)
                np_vgg_fea.append(clus_feat)
            all_vgg.append(np_vgg_fea)

            for each_set in Batch_set:
                surv_time_train.append(each_set[3])
                status_train.append(each_set[2])

            surv_time_train = np.asarray(surv_time_train)
            status_train = np.asarray(status_train)
            np_cls_num = np.asarray(cur_cluster)

            sample = {'feat': all_vgg[0], 'mask':mask, 'time': surv_time_train[0], 'status':status_train[0], 'cluster_num': np_cls_num}
            if self.transform:
                sample = self.transform(sample)
            return sample
        except Exception as e:
            print(f"[ERROR] Failed loading {img_path}: {e}")
            raise

class ToTensor(object):
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
    def __call__(self, sample):
        image, time, status = sample['feat'], sample['time'], sample['status']
        return {'feat': [torch.from_numpy(image[i]) for i in range(self.cluster_num)],
                'time': torch.FloatTensor([time]),
                'status': torch.FloatTensor([status]),
                'mask': torch.from_numpy(sample['mask']),
                'cluster_num': torch.from_numpy(sample['cluster_num'])
                }
