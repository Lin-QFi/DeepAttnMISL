"""
Define pytorch dataloader for DeepAttnMISL
"""
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def set_mil_seed(seed):   
    global GLOBAL_MIL_SEED
    GLOBAL_MIL_SEED = seed

def get_mil_seed():       
    global GLOBAL_MIL_SEED
    return GLOBAL_MIL_SEED if GLOBAL_MIL_SEED is not None else 666

def worker_init_fn(worker_id):  
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
                    from sklearn.model_selection import train_test_split
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
            all_resnet = []
            resnet_clus = [[] for i in range(self.cluster_num)]
            npz_file = np.load(img_path)
            # 只支持 resnet_features
            cur_resnet = npz_file['resnet_features']
            resnet_feat_dim = 2048
            cur_patient = npz_file['pid']
            cur_time = npz_file['time']
            cur_status = npz_file['status']
            cur_path = npz_file['img_path']
            cur_cluster = npz_file['cluster_num']

            # ==== 新增：读取临床参数 =====
            cur_clinical_param = None
            if 'clinical_param' in npz_file:
                cur_clinical_param = npz_file['clinical_param']
            # ===========================

            for id, each_patch_cls in enumerate(cur_cluster):
                resnet_clus[each_patch_cls].append(cur_resnet[id])

            Batch_set.append((cur_resnet, cur_patient, cur_status, cur_time, cur_cluster))
            np_resnet_fea = []
            mask = np.ones(self.cluster_num, dtype=np.float32)
            for i in range(self.cluster_num):
                if len(resnet_clus[i]) == 0:
                    clus_feat = np.zeros((1, resnet_feat_dim), dtype=np.float32)
                    mask[i] = 0
                else:
                    if self.random:
                        curr_feat = resnet_clus[i]
                        ind = np.arange(len(curr_feat))
                        np.random.seed(get_mil_seed())
                        np.random.shuffle(ind)
                        clus_feat = np.asarray([curr_feat[i] for i in ind])
                    else:
                        clus_feat = np.asarray(resnet_clus[i])
                clus_feat = np.swapaxes(clus_feat, 1, 0)
                clus_feat = np.expand_dims(clus_feat, 1)
                np_resnet_fea.append(clus_feat)
            all_resnet.append(np_resnet_fea)

            for each_set in Batch_set:
                surv_time_train.append(each_set[3])
                status_train.append(each_set[2])

            surv_time_train = np.asarray(surv_time_train)
            status_train = np.asarray(status_train)
            np_cls_num = np.asarray(cur_cluster)

            sample = {
                'feat': all_resnet[0],
                'mask': mask,
                'time': surv_time_train[0],
                'status': status_train[0],
                'cluster_num': np_cls_num,
                'clinical_param': cur_clinical_param # 新增
            }
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
        out = {
            'feat': [torch.from_numpy(image[i]) for i in range(self.cluster_num)],
            'time': torch.FloatTensor([time]),
            'status': torch.FloatTensor([status]),
            'mask': torch.from_numpy(sample['mask']),
            'cluster_num': torch.from_numpy(sample['cluster_num'])
        }
        # ===== 新增：临床参数Tensor化 =====
        if 'clinical_param' in sample and sample['clinical_param'] is not None:
            out['clinical_param'] = torch.from_numpy(sample['clinical_param']).float()
        return out
