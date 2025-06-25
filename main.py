import gc
import torch
import numpy as np
from MIL_dataloader import MIL_dataloader
from tqdm import tqdm
from utils.surv_utils import cox_log_rank, CIndex_lifeline
from DeepAttnMISL_model import DeepAttnMIL_Surv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import os
from utils.Early_Stopping import EarlyStopping
from sklearn.model_selection import KFold
import argparse
import random
from MIL_dataloader import set_mil_seed

def set_seed(seed=666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_mil_seed(seed)

parser = argparse.ArgumentParser(description='DeepAttnMISL')
parser.add_argument('--nfolds', type=int, default=5, help='number of folds for cross-validation')
parser.add_argument('--cluster_num', type=int, default=10, help='cluster number')
parser.add_argument('--feat_path', type=str, default='/media/zsly/2EF669DFF669A833/DeepAttnMISL/each_patient/kmeans/cluster_num_10', help='deep features directory')
parser.add_argument('--img_label_path', type=str, default='/media/zsly/2EF669DFF669A833/DeepAttnMISL/data/label_csv/all_patch_expandedlabels.csv')
parser.add_argument('--batch_size', type=int, default=1, help='has to be 1')
parser.add_argument('--nepochs', type=int, default=100, help='The maximum number of epochs to train')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate (default: 1e-4)')
parser.add_argument('--seed', default=666, type=int, help='random seed')
parser.add_argument('--clinical_dim', type=int, default=27, help='dimension of clinical_param vector')
args = parser.parse_args()

def _neg_partial_log(prediction, T, E):
    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]
    train_R = torch.FloatTensor(R_matrix_train).cuda()
    train_ystatus = torch.FloatTensor(E).cuda()
    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)
    return loss_nn

def train_epoch(epoch, model, optimizer, trainloader, cluster_num, clinical_dim, measure=1, verbose=1):
    model.train()
    lbl_pred_all = None
    lbl_pred_each = None
    survtime_all = []
    status_all = []
    iter = 0
    gc.collect()
    loss_nn_all = []
    tbar = tqdm(trainloader, desc='\r')
    for i_batch, sampled_batch in enumerate(tbar):
        X, survtime, lbl, mask, clinical_param = (
            sampled_batch['feat'],
            sampled_batch['time'],
            sampled_batch['status'],
            sampled_batch['mask'],
            sampled_batch['clinical_param']
        )
        if len(X) < cluster_num:
            print(f"Warning: cluster_num={cluster_num}, but only {len(X)} clusters found for this sample. Skipping sample.")
            continue
        graph = [X[i].cuda() for i in range(cluster_num)]
        lbl = lbl.cuda()
        masked_cls = mask.cuda()
        clinical_param = clinical_param.cuda().unsqueeze(0) if clinical_param.dim() == 1 else clinical_param.cuda()
        lbl_pred = model(graph, masked_cls, clinical_param)
        time = survtime.data.cpu().numpy()
        status = lbl.data.cpu().numpy()
        time = np.squeeze(time)
        status = np.squeeze(status)
        pred_score = lbl_pred.detach().cpu().numpy()
        survtime_all.append(time/30.0)
        status_all.append(status)
        if i_batch == 0:
            lbl_pred_all = lbl_pred
            survtime_torch = survtime
            lbl_torch = lbl
        if iter == 0:
            lbl_pred_each = lbl_pred
        else:
            lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
            lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred])
            lbl_torch = torch.cat([lbl_torch, lbl])
            survtime_torch = torch.cat([survtime_torch, survtime])
        iter += 1
        if iter % 16 == 0 or i_batch == len(trainloader)-1:
            survtime_all = np.asarray(survtime_all)
            status_all = np.asarray(status_all)
            if np.max(status_all) == 0:
                print("encounter no death in a batch, skip")
                lbl_pred_each = None
                survtime_all = []
                status_all = []
                iter = 0
                continue
            optimizer.zero_grad()
            loss_surv = _neg_partial_log(lbl_pred_each, survtime_all, status_all)
            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum()
            loss = loss_surv + 1e-5 * l1_reg
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            lbl_pred_each = None
            survtime_all = []
            status_all = []
            loss_nn_all.append(loss.data.item())
            iter = 0
            gc.collect()
    mean_train_loss = np.mean(loss_nn_all) if len(loss_nn_all) > 0 else None
    if measure:
        pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survtime_torch)
        c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survtime_torch)
        if verbose > 0:
            print("\nEpoch: {}, loss_nn: {}".format(epoch, mean_train_loss))
            print('\n[Training]\t loss (nn):{:.4f}'.format(mean_train_loss),
                  'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    return mean_train_loss

def save_metrics_to_csv(metrics, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

def train(train_path, test_path, model_save_path, num_epochs, lr, cluster_num=10,
          clinical_dim=27, fold=0, total_folds=5, script_dir=None):
    # Get train/val loader
    trainloader, valloader = MIL_dataloader(data_path=train_path, cluster_num=cluster_num, train=True).get_loader()
    # Get test loader
    testloader = MIL_dataloader(data_path=test_path, cluster_num=cluster_num, train=False).get_loader()
    model = DeepAttnMIL_Surv(cluster_num=cluster_num, clinical_dim=clinical_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-4)
    early_stopping = EarlyStopping(model_path=model_save_path, patience=15, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    save_epoch = range(10, 100, 5)
    val_ci_list = []
    val_losses = []
    metrics = []

    # 确定保存路径
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = f"{total_folds}folds_{cluster_num}cluster_num"
    csv_filename = f"{folder_name}_fold_{fold}.csv"
    csv_dir = os.path.join(script_dir, folder_name)
    csv_path = os.path.join(csv_dir, csv_filename)

    for epoch in range(num_epochs):
        train_loss = train_epoch(epoch, model, optimizer, trainloader, cluster_num, clinical_dim)
        valid_loss, val_ci, val_pvalue = prediction(model, valloader, cluster_num, clinical_dim, return_pvalue=True)
        scheduler.step(valid_loss)
        val_losses.append(valid_loss)
        early_stopping(valid_loss, model)
        # 记录本epoch指标
        metrics.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': valid_loss, 'c_index': val_ci, 'p_value': val_pvalue})
        save_metrics_to_csv(metrics, csv_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch in save_epoch:
            val_ci_list.append(val_ci)
            print('saving epoch in {}, vali loss: {}, val ci:{}'.format(epoch, valid_loss, val_ci))
            torch.save(model.state_dict(), model_save_path.replace('.pth', '_epoch_{}.pth'.format(epoch)))

    model_test = DeepAttnMIL_Surv(cluster_num=cluster_num, clinical_dim=clinical_dim).cuda()
    model_test.load_state_dict(torch.load(model_save_path))
    _, c_index, _ = prediction(model_test, testloader, cluster_num, clinical_dim, testing=True, return_pvalue=True)
    return c_index

def prediction(model, queryloader, cluster_num, clinical_dim, testing=False, return_pvalue=False):
    model.eval()
    lbl_pred_all = None
    status_all = []
    survtime_all = []
    iter = 0
    tbar = tqdm(queryloader, desc='\r')
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(tbar):
            X, survtime, lbl, cls_num, mask, clinical_param = (
                sampled_batch['feat'],
                sampled_batch['time'],
                sampled_batch['status'],
                sampled_batch['cluster_num'],
                sampled_batch['mask'],
                sampled_batch['clinical_param']
            )
            if len(X) < cluster_num:
                print(f"Warning: cluster_num={cluster_num}, but only {len(X)} clusters found for this sample. Skipping sample.")
                continue
            graph = [X[i].cuda() for i in range(cluster_num)]
            lbl = lbl.cuda()
            clinical_param = clinical_param.cuda().unsqueeze(0) if clinical_param.dim() == 1 else clinical_param.cuda()
            time = survtime.data.cpu().numpy()
            status = lbl.data.cpu().numpy()
            time = np.squeeze(time)
            status = np.squeeze(status)
            survtime_all.append(time/30.0)
            status_all.append(status)
            lbl_pred = model(graph, mask.cuda(), clinical_param)
            if iter == 0:
                lbl_pred_all = lbl_pred
                survtime_torch = survtime
                lbl_torch = lbl
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                lbl_torch = torch.cat([lbl_torch, lbl])
                survtime_torch = torch.cat([survtime_torch, survtime])
            iter += 1
    survtime_all = np.asarray(survtime_all)
    status_all = np.asarray(status_all)
    loss_surv = _neg_partial_log(lbl_pred_all, survtime_all, status_all)
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    loss = loss_surv + 3e-5 * l1_reg
    print("\nval_loss_nn: %.4f, L1: %.4f" % (loss_surv, 1e-5 * l1_reg))
    pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survtime_torch)
    c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survtime_torch)
    if not testing:
        print('\n[val]\t loss (nn):{:.4f}'.format(loss.data.item()),
                      'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    else:
        print('\n[testing]\t loss (nn):{:.4f}'.format(loss.data.item()),
              'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    if return_pvalue:
        return loss.data.item(), c_index, pvalue_pred
    else:
        return loss.data.item(), c_index

if __name__ == '__main__':
    set_seed(args.seed)
    cluster_num = args.cluster_num
    nfolds = args.nfolds
    feat_path = args.feat_path
    img_label_path = args.img_label_path
    batch_size = args.batch_size
    num_epochs = args.nepochs
    lr = args.lr
    clinical_dim = args.clinical_dim

    print(f"Using feature path: {feat_path}")
    print(f"Using label path: {img_label_path}")
    print(f"Cluster num: {cluster_num}, nfolds: {nfolds}, lr: {lr}, epochs: {num_epochs}, seed: {args.seed}, clinical_dim: {clinical_dim}")

    all_paths = pd.read_csv(img_label_path)
    surv = all_paths['surv']
    status = all_paths['status'].tolist()
    pid = all_paths['pid'].tolist()
    uniq_pid = np.unique(pid)
    uniq_st = []
    for each_pid in uniq_pid:
        temp = pid.index(each_pid)
        uniq_st.append(status[temp])

    # ONLY KEEP PIDS THAT HAVE NPZ
    import os
    feat_npz_dir = feat_path
    exist_npz_ids = set(os.path.splitext(os.path.basename(f))[0] for f in os.listdir(feat_npz_dir) if f.endswith('.npz'))
    uniq_pid = np.array([pid for pid in uniq_pid if str(pid) in exist_npz_ids])
    uniq_st = np.array([status[pid.index(each_pid)] for each_pid in uniq_pid])

    kf = KFold(n_splits=nfolds, random_state=args.seed, shuffle=True)
    testci = []
    fold = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for train_index, test_index in kf.split(range(len(uniq_st))):
        print("Now training fold:{}".format(fold))
        test_pid = [uniq_pid[i] for i in test_index]
        print('testing pid', len(test_pid))
        train_val_npz = [str(uniq_pid[i])+'.npz' for i in train_index]
        test_npz = [str(uniq_pid[i])+'.npz' for i in test_index]
        train_val_patients_pca = [os.path.join(feat_path, each_path) for each_path in train_val_npz]
        test_patients_pca = [os.path.join(feat_path, each_path) for each_path in test_npz]
        print('training pid', len(train_val_patients_pca))
        print('testing pid', len(test_pid))
        os.makedirs('./saved_model', exist_ok=True)
        model_save_path = './saved_model/NLST_model_fold_{}_c_{}.pth'.format(fold, cluster_num)
        test_ci = train(
            train_val_patients_pca, test_patients_pca, model_save_path,
            num_epochs=num_epochs, lr=lr, cluster_num=cluster_num,
            clinical_dim=clinical_dim,
            fold=fold, total_folds=nfolds, script_dir=script_dir
        )
        testci.append(test_ci)
        fold += 1
    print(testci)
    print(np.mean(testci))
