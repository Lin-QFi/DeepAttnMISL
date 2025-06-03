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
from sklearn.model_selection import KFold, train_test_split # 修改：引入train_test_split
import argparse
from pathlib import Path
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
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate (default: 1e-4)')

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

def prediction(model, queryloader, testing=False):
    model.eval()
    lbl_pred_all = None
    status_all = []
    survtime_all = []
    iter = 0
    tbar = tqdm(queryloader, desc='\r')
    for i_batch, sampled_batch in enumerate(tbar):
        X, survtime, lbl, cls_num, mask = sampled_batch['feat'], sampled_batch['time'], sampled_batch['status'], sampled_batch['cluster_num'], sampled_batch['mask']
        if len(X) < cluster_num:
            print(f"Warning: cluster_num={cluster_num}, but only {len(X)} clusters found for this sample. Skipping sample.")
            continue
        graph = [X[i].cuda() for i in range(cluster_num)]
        lbl = lbl.cuda()
        time = survtime.data.cpu().numpy()
        status = lbl.data.cpu().numpy()
        time = np.squeeze(time)
        status = np.squeeze(status)
        survtime_all.append(time/30.0)
        status_all.append(status)
        lbl_pred = model(graph, mask.cuda())
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
    print("Validation statuses:", status_all)
    print("Validation survival times:", survtime_all)
    loss_surv = _neg_partial_log(lbl_pred_all, survtime_all, status_all)
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    loss = loss_surv + 1e-5 * l1_reg
    print("\nval_loss_nn: %.4f, L1: %.4f" % (loss_surv, 1e-5 * l1_reg))
    pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survtime_torch)
    c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survtime_torch)
    if not testing:
        print('\n[val]\t loss (nn):{:.4f}'.format(loss.data.item()),
                      'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    else:
        print('\n[testing]\t loss (nn):{:.4f}'.format(loss.data.item()),
              'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    return loss.data.item(), c_index, pvalue_pred

def train_epoch(epoch, model, optimizer, trainloader,  measure=1, verbose=1):
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
        X, survtime, lbl, mask = sampled_batch['feat'], sampled_batch['time'], sampled_batch['status'], sampled_batch['mask']
        if len(X) < cluster_num:
            print(f"Warning: cluster_num={cluster_num}, but only {len(X)} clusters found for this sample. Skipping sample.")
            continue
        graph = [X[i].cuda() for i in range(cluster_num)]
        lbl = lbl.cuda()
        masked_cls = mask.cuda()
        lbl_pred = model(graph, masked_cls)
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
    if measure:
        pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survtime_torch)
        c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survtime_torch)
        if verbose > 0:
            print("\nEpoch: {}, loss_nn: {}".format(epoch, np.mean(loss_nn_all)))
            print('\n[Training]\t loss (nn):{:.4f}'.format(np.mean(loss_nn_all)),
                  'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))

def save_metrics_to_csv(metrics, csv_path):
    import pandas as pd
    df = pd.DataFrame(metrics)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

def train(train_path, test_path, model_save_path, num_epochs, lr, cluster_num=10, fold=None, total_folds=None):
    model = DeepAttnMIL_Surv(cluster_num=cluster_num).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-4)
    Data = MIL_dataloader(data_path=train_path, cluster_num = cluster_num, train=True)
    trainloader, valloader = Data.get_loader()
    TestData = MIL_dataloader(test_path, cluster_num=cluster_num, train=False)
    testloader = TestData.get_loader()
    early_stopping = EarlyStopping(model_path=model_save_path,patience=15, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    save_epoch = range(10, 100, 5)
    val_ci_list = []
    val_losses = []
    metrics = []
    for epoch in range(num_epochs):
        train_epoch(epoch, model, optimizer, trainloader)
        valid_loss, val_ci, val_pvalue = prediction(model, valloader)
        scheduler.step(valid_loss)
        val_losses.append(valid_loss)
        metrics.append({'epoch': epoch, 'loss': valid_loss, 'c_index': val_ci, 'p_value': val_pvalue})
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch in save_epoch:
            val_ci_list.append(val_ci)
            print('saving epoch in {}, vali loss: {}, val ci:{}'.format(epoch, valid_loss, val_ci))
            torch.save(model.state_dict(), model_save_path.replace('.pth', '_epoch_{}.pth'.format(epoch)))
    if total_folds is not None:
        folder_name = f"{total_folds}folds_{cluster_num}cluster_num"
        if fold is not None:
            csv_filename = f"{folder_name}_fold_{fold}.csv"
        else:
            csv_filename = f"{folder_name}.csv"
    else:
        folder_name = f"folds_{cluster_num}cluster_num"
        if fold is not None:
            csv_filename = f"{folder_name}_fold_{fold}.csv"
        else:
            csv_filename = f"{folder_name}.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(script_dir, folder_name)
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, csv_filename)
    save_metrics_to_csv(metrics, csv_path)
    model_test = DeepAttnMIL_Surv(cluster_num=cluster_num).cuda()
    model_test.load_state_dict(torch.load(model_save_path))
    _, c_index, _ = prediction(model_test, testloader, testing=True)
    return c_index, model_save_path # 修改：返回模型路径

if __name__ == '__main__':
    seed = 666
    set_seed(seed)
    args = parser.parse_args()
    cluster_num = args.cluster_num

    args.img_label_path = '/media/zsly/2EF669DFF669A833/DeepAttnMISL/data/label_csv/all_patch_expandedlabels.csv'
    print(f"使用标签文件: {args.img_label_path}")

    output_root = '/media/zsly/2EF669DFF669A833/DeepAttnMISL/each_patient/kmeans'
    generate_npz_path = '/media/zsly/2EF669DFF669A833/DeepAttnMISL/generate_npz.py'
    if args.feat_path:
        feat_path_exp = str(Path(args.feat_path))
    else:
        feat_path = os.path.join(output_root, f'cluster_num_{cluster_num}')
        feat_path_exp = feat_path
    os.makedirs(feat_path_exp, exist_ok=True)
    npz_files = list(Path(feat_path_exp).glob("*.npz"))
    if len(npz_files) < 10:
        print(f"自动调用 generate_npz.py 生成特征，cluster_num={cluster_num} ...")
        cmd = f"python {generate_npz_path} --cluster_num {cluster_num} --output_root {output_root}"
        os.system(cmd)
        npz_files = list(Path(feat_path_exp).glob("*.npz"))
    args.feat_path = feat_path_exp
    print(f"Using feature path: {args.feat_path}")

    img_label_path = args.img_label_path
    batch_size = args.batch_size
    num_epochs = args.nepochs
    lr = args.lr

    all_paths = pd.read_csv(img_label_path)
    surv = all_paths['surv']
    status = all_paths['status'].tolist()
    pid = all_paths['pid'].tolist()
    uniq_pid = np.unique(pid)  # unique patients id

    # ========== 修改开始：先分出10% holdout ==========
    uniq_pid = np.array(uniq_pid)
    uniq_st = np.array([status[pid.index(each_pid)] for each_pid in uniq_pid])

    # 分出10% holdout set
    train_pid, holdout_pid, train_st, holdout_st = train_test_split(
        uniq_pid, uniq_st, test_size=0.1, random_state=666, stratify=uniq_st
    )
    print(f"Holdout set patients: {len(holdout_pid)}")
    # ========== 修改结束 ==========

    # ========== 修改开始：5折交叉验证在90%上 ==========
    kf = KFold(n_splits=args.nfolds, random_state=666, shuffle=True)
    n_folds = kf.get_n_splits()
    train_pid_ind = range(len(train_pid))
    uniq_pid_for_kfold = train_pid
    uniq_st_for_kfold = train_st

    def has_event(test_index, uniq_st):
        return any(uniq_st[i] == 1 for i in test_index)

    # 只在train_pid上做5折
    max_retries = 50
    retry = 0
    valid_splits = None
    while retry < max_retries:
        retry += 1
        splits = list(kf.split(train_pid_ind))
        if all(has_event(test_index, uniq_st_for_kfold) for _, test_index in splits):
            print(f"Found valid fold split on attempt {retry}")
            valid_splits = splits
            break
        else:
            print(f"Retrying fold split attempt {retry}...")
    if valid_splits is None:
        raise RuntimeError("Failed to find valid KFold splits with events in every validation set.")
    # ========== 修改结束 ==========

    testci = []
    fold = 0
    model_paths = [] # 修改：保存每折模型路径
    for train_index, test_index in valid_splits:
        print("Now training fold:{}".format(fold))
        train_val_npz = [str(uniq_pid_for_kfold[i]) + '.npz' for i in train_index]
        test_npz = [str(uniq_pid_for_kfold[i]) + '.npz' for i in test_index]
        train_val_patients_pca = [os.path.join(feat_path_exp, each_path) for each_path in train_val_npz]
        test_patients_pca = [os.path.join(feat_path_exp, each_path) for each_path in test_npz]
        print('training pid', len(train_val_patients_pca))
        print('testing pid', len(test_npz))
        os.makedirs('./saved_model', exist_ok=True)
        model_save_path = './saved_model/NLST_model_fold_{}_c_{}.pth'.format(fold, cluster_num)
        test_ci, model_path = train(
            train_val_patients_pca, test_patients_pca, model_save_path,
            num_epochs=num_epochs, lr=lr, cluster_num=cluster_num,
            fold=fold, total_folds=n_folds
        )
        testci.append(test_ci)
        model_paths.append(model_path)
        fold += 1
    print(testci)
    print(np.mean(testci))

    # ========== 新增：在holdout set评估前，评估整个训练集 ==========
    print("\n========== Evaluating on the full training set ==========")
    train_all_npz = [os.path.join(feat_path_exp, str(pid)+'.npz') for pid in train_pid]
    trainAll_TestData = MIL_dataloader(train_all_npz, cluster_num=cluster_num, train=False)
    trainAll_loader = trainAll_TestData.get_loader()
    
    # 1. 用每一折模型评估训练集
    trainall_cindex_list = []
    for model_path in model_paths:
        model_test = DeepAttnMIL_Surv(cluster_num=cluster_num).cuda()
        model_test.load_state_dict(torch.load(model_path))
        _, trainall_c_index, _ = prediction(model_test, trainAll_loader, testing=True)
        trainall_cindex_list.append(trainall_c_index)
    print("Train set c-index for each fold model:", trainall_cindex_list)
    print("Train set c-index mean:", np.mean(trainall_cindex_list))
    
    # 2. 用最后一折模型评估训练集
    print("\n========== Evaluating on training set using the last fold's model ==========")
    model_lastfold = DeepAttnMIL_Surv(cluster_num=cluster_num).cuda()
    model_lastfold.load_state_dict(torch.load(model_paths[-1]))
    _, trainall_c_index_last, _ = prediction(model_lastfold, trainAll_loader, testing=True)
    print("Train set c-index for the last fold model:", trainall_c_index_last)
    # ========== 新增结束 ==========
    # ========== 修改开始：在holdout set上分别评估 ==========
    print("\n========== Evaluating on 10% holdout set ==========")
    holdout_npz = [os.path.join(feat_path_exp, str(pid)+'.npz') for pid in holdout_pid]
    holdout_TestData = MIL_dataloader(holdout_npz, cluster_num=cluster_num, train=False)
    holdout_loader = holdout_TestData.get_loader()

    # 1. 用每一折模型评估 holdout set
    holdout_cindex_list = []
    for model_path in model_paths:
        model_test = DeepAttnMIL_Surv(cluster_num=cluster_num).cuda()
        model_test.load_state_dict(torch.load(model_path))
        _, holdout_c_index, _ = prediction(model_test, holdout_loader, testing=True)
        holdout_cindex_list.append(holdout_c_index)
    print("Holdout set c-index for each fold model:", holdout_cindex_list)
    print("Holdout set c-index mean:", np.mean(holdout_cindex_list))

    # 2. 用最后一折模型评估 holdout set
    print("\n========== Evaluating on holdout set using the last fold's model ==========")
    last_fold_model_path = model_paths[-1]
    model_lastfold = DeepAttnMIL_Surv(cluster_num=cluster_num).cuda()
    model_lastfold.load_state_dict(torch.load(last_fold_model_path))
    _, holdout_c_index_last, _ = prediction(model_lastfold, holdout_loader, testing=True)
    print("Holdout set c-index for the last fold model:", holdout_c_index_last)
    # ========== 修改结束 ==========
