import os
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# ========== 参数设置 ==========
parser = argparse.ArgumentParser()
parser.add_argument('--cluster_num', type=int, default=10, help='kmeans n_clusters')
parser.add_argument('--output_root', type=str, default='/media/zsly/2EF669DFF669A833/DeepAttnMISL/each_patient/kmeans', help='聚类特征根目录')
args = parser.parse_args()

patch_root = Path("/media/zsly/2EF669DFF669A833/DeepAttnMISL/data/patches")
output_npz_dir = Path(args.output_root) / f"cluster_num_{args.cluster_num}"
output_npz_dir.mkdir(parents=True, exist_ok=True)
clinical_file = Path("/media/zsly/2EF669DFF669A833/DeepAttnMISL/data/folder_names_ID.xlsx")

# ========== 加载临床数据 ==========
clinical_df = pd.read_excel(clinical_file)
clinical_df['patient_ID'] = clinical_df['patient_ID'].astype(str).str.strip()

# ====== 明确需要用的列 ======
NUMERIC_COLS = ['age', 'BMI', 'CEA', 'CA199', 'CA125', 'CA153', 'AFP']
BINARY_COLS = ['sex', 'ganzhuanyi', 'feizhuanyi', 'qitazhuanyi', 'maiguanneiaishuan', 'shenjingshujinrun']
MULTI_CAT_COLS = ['weizhi', 'fenxing', 'fenhua']
ALL_COLS = NUMERIC_COLS + BINARY_COLS + MULTI_CAT_COLS

# ========== 数值标准化(全队列) ==========
num_mask = clinical_df[NUMERIC_COLS].notna().all(axis=1)
num_mean = clinical_df.loc[num_mask, NUMERIC_COLS].mean()
num_std = clinical_df.loc[num_mask, NUMERIC_COLS].std().replace(0, 1)  # 避免除0

# ========== 多分类变量one-hot编码(获取全类别) ==========
multi_cat_uniques = {col: sorted(clinical_df[col].dropna().unique()) for col in MULTI_CAT_COLS}
multi_cat_value2idx = {col: {v: i for i, v in enumerate(vals)} for col, vals in multi_cat_uniques.items()}
multi_cat_dim = sum(len(v) for v in multi_cat_uniques.values())

# 保存标准化参数以便推理时复现
norm_param_save_path = output_npz_dir / "clinical_norm_params.npz"
np.savez(
    norm_param_save_path,
    num_mean=num_mean.values.astype(np.float32),
    num_std=num_std.values.astype(np.float32),
    num_names=np.array(NUMERIC_COLS),
    multi_cat_uniques={k: np.array(v) for k, v in multi_cat_uniques.items()}
)

# ========== 设备设置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载 ResNet50 ==========
weights = ResNet50_Weights.IMAGENET1K_V1
resnet = resnet50(weights=weights)
resnet.fc = torch.nn.Identity()  # 去掉全连接层
resnet = resnet.to(device).eval()

# ========== 图像预处理 ==========
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========== 只遍历excel里的病人 ==========
for patient_id in tqdm(clinical_df['patient_ID'], desc="生成 .npz"):
    patient_patch_dir = patch_root / patient_id
    if not patient_patch_dir.is_dir():
        print(f"❌ {patient_id} 无patch文件夹")
        continue

    patch_paths = sorted(patient_patch_dir.glob("*.jpg"))
    if len(patch_paths) == 0:
        print(f"❌ {patient_id} patch文件夹为空")
        continue

    features = []
    img_paths_str = []

    for path in patch_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(img_tensor).cpu().numpy().squeeze()  # 2048维
        features.append(feat)
        img_paths_str.append(str(path))

    features = np.vstack(features)

    cluster_model = KMeans(n_clusters=args.cluster_num, random_state=42)
    cluster_labels = cluster_model.fit_predict(features)

    # 匹配临床数据
    match = clinical_df.loc[clinical_df["patient_ID"] == patient_id]
    if match.empty:
        print(f"❌ {patient_id} 无临床信息")
        continue

    time = match["followupdays"].values[0]
    status_str = match["demographicvitalstatus"].values[0]
    status = 1 if str(status_str).strip().lower() == 'dead' else 0

    # ========== 提取/处理临床参数 ==========
    try:
        # 数值型标准化
        numeric = ((match[NUMERIC_COLS].values[0] - num_mean.values) / num_std.values).astype(np.float32)
        # 二分类直接取
        binary = match[BINARY_COLS].values[0].astype(np.float32)
        # 多分类one-hot
        multi_cat_onehot = []
        for col in MULTI_CAT_COLS:
            val = match[col].values[0]
            onehot = np.zeros(len(multi_cat_uniques[col]), dtype=np.float32)
            if pd.notnull(val):
                idx = multi_cat_value2idx[col].get(val, None)
                if idx is not None:
                    onehot[idx] = 1.0
            multi_cat_onehot.append(onehot)
        multi_cat_onehot = np.concatenate(multi_cat_onehot, axis=0)
        clinical_param_vector = np.concatenate([numeric, binary, multi_cat_onehot], axis=0).astype(np.float32)
    except Exception as e:
        print(f"❌ {patient_id} 临床参数提取失败: {e}")
        continue

    # ========== 保存 =============
    np.savez(
        output_npz_dir / f"{patient_id}.npz",
        resnet_features=features.astype(np.float32),
        pid=patient_id,
        time=float(time),
        status=int(status),
        img_path=np.array(img_paths_str),
        cluster_num=cluster_labels.astype(np.int32),
        clinical_param=clinical_param_vector
    )

    print(f"✅ 生成成功：{patient_id}.npz, 临床参数shape: {clinical_param_vector.shape}")
