import os
import numpy as np
import torch
from torchvision.models import vgg16, VGG16_Weights
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

# 设置VGG权重路径（需提前下载并放到此处）
os.environ['TORCH_HOME'] = str(Path('~/LQF/vgg').expanduser())

patch_root = Path("/media/zsly/2EF669DFF669A833/DeepAttnMISL/data/patches")
output_npz_dir = Path(args.output_root) / f"cluster_num_{args.cluster_num}"
output_npz_dir.mkdir(parents=True, exist_ok=True)
clinical_file = Path("/media/zsly/2EF669DFF669A833/DeepAttnMISL/data/folder_names_ID.xlsx")

# ========== 加载临床数据 ==========
clinical_df = pd.read_excel(clinical_file)
clinical_df['patient_ID'] = clinical_df['patient_ID'].astype(str).str.strip()

# ========== 图像预处理 ==========
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========== 加载 VGG ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = VGG16_Weights.IMAGENET1K_V1
vgg = vgg16(weights=weights)
vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-3])
vgg = vgg.to(device).eval()

# ========== 遍历每个 patient ==========
for patient_dir in tqdm(sorted(patch_root.iterdir()), desc="生成 .npz"):
    if not patient_dir.is_dir():
        continue

    patient_id = patient_dir.name.strip()
    patch_paths = sorted(patient_dir.glob("*.jpg"))
    if len(patch_paths) == 0:
        print(f"❌ {patient_id} 无 patch 图像")
        continue

    features = []
    img_paths_str = []

    for path in patch_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = vgg(img_tensor).cpu().numpy().squeeze()
        features.append(feat)
        img_paths_str.append(str(path))

    vgg_features = np.vstack(features)

    cluster_model = KMeans(n_clusters=args.cluster_num, random_state=42)
    cluster_labels = cluster_model.fit_predict(vgg_features)

    # 匹配临床数据
    match = clinical_df.loc[clinical_df["patient_ID"] == patient_id]
    if match.empty:
        print(f"❌ {patient_id} 无临床信息")
        continue

    time = match["followupdays"].values[0]
    status_str = match["demographicvitalstatus"].values[0]
    status = 1 if str(status_str).strip().lower() == 'dead' else 0

    # ========== 保存 =============
    np.savez(
        output_npz_dir / f"{patient_id}.npz",
        vgg_features=vgg_features.astype(np.float32),
        pid=patient_id,
        time=float(time),
        status=int(status),
        img_path=np.array(img_paths_str),
        cluster_num=cluster_labels.astype(np.int32),
    )

    print(f"✅ 生成成功：{patient_id}.npz")
