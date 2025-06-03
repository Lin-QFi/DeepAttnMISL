import os
import random
import openslide
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

random.seed(42) # 随机数固定
np.random.seed(42) # 随机数固定

# 设置目录路径
tiff_dirs = [
    Path("/media/zsly/2EF669DFF669A833/all_tiff/PM0"),
    Path("/media/zsly/2EF669DFF669A833/all_tiff/PM1"),
]
patch_dir = Path("/media/zsly/2EF669DFF669A833/DeepAttnMISL/data/patches")
folder_names_file = Path("/media/zsly/2EF669DFF669A833/DeepAttnMISL/data/folder_names_ID.xlsx")
debug_dir = Path("/media/zsly/2EF669DFF669A833/DeepAttnMISL/data/debug_invalid_patches")
debug_dir.mkdir(parents=True, exist_ok=True)

# 参数配置
PATCH_SIZE = 224
MAX_PATCHES = 100
TISSUE_THRESHOLD = 0.8  # 白色像素占比大于此值视为背景

# 读取映射表
folder_names_data = pd.read_excel(folder_names_file)
valid_folder_names = set(folder_names_data['Folder Name'].astype(str))

def is_valid_patch(patch, white_thresh=TISSUE_THRESHOLD, min_saturation=15, min_hue_var=2, min_std=10):
    arr = np.array(patch)[:, :, :3]
    # 白色像素占比
    white_ratio = np.mean(np.all(arr > 220, axis=2))
    if white_ratio >= white_thresh:
        return False, {'reason': 'high_white', 'white_ratio': white_ratio}
    # 灰度方差
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray_std = gray.std()
    if gray_std < min_std:
        return False, {'reason': 'low_gray_std', 'gray_std': gray_std}
    # HSV饱和度均值
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat_mean = hsv[:, :, 1].mean()
    if sat_mean < min_saturation:
        return False, {'reason': 'low_saturation', 'sat_mean': sat_mean}
    # HSV色调方差
    hue_var = hsv[:, :, 0].var()
    if hue_var < min_hue_var:
        return False, {'reason': 'low_hue_var', 'hue_var': hue_var}
    return True, {}

def get_tissue_mask(slide, level=2, threshold=210):
    if level >= slide.level_count:
        level = slide.level_count - 1
    thumbnail = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
    thumbnail_np = np.array(thumbnail)
    gray = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask.astype(bool), slide.level_downsamples[level]

# 收集所有需要处理的tiff文件
all_tiff_files = []
for tiff_dir in tiff_dirs:
    for fname in os.listdir(tiff_dir):
        fpath = tiff_dir / fname
        # 只处理有效表格内的tiff文件
        if fpath.is_file() and fname in valid_folder_names:
            all_tiff_files.append((fname, fpath))

for tiff_file, tiff_path in tqdm(all_tiff_files, desc="处理WSI样本"):
    patient_id = folder_names_data.loc[folder_names_data['Folder Name'] == tiff_file, 'patient_ID'].values
    if len(patient_id) == 0:
        tqdm.write(f"❌ 未找到文件 {tiff_file} 的 patient_ID")
        continue
    patient_id = str(patient_id[0])
    patient_patch_dir = patch_dir / patient_id
    patient_patch_dir.mkdir(parents=True, exist_ok=True)
    debug_patient_dir = debug_dir / patient_id
    debug_patient_dir.mkdir(parents=True, exist_ok=True)
    try:
        slide = openslide.OpenSlide(str(tiff_path))
    except Exception as e:
        tqdm.write(f"❌ 无法读取 {tiff_file}: {e}")
        continue

    mask, downsample = get_tissue_mask(slide)
    tissue_coords = np.argwhere(mask)
    tqdm.write(f"{tiff_file}: 组织区域像素点数 = {len(tissue_coords)}")
    if len(tissue_coords) == 0:
        tqdm.write(f"⚠️ WSI {tiff_file} 没有组织区域")
        continue

    total_saved = 0
    total_invalid = 0
    max_attempts = 1000
    attempts = 0
    while total_saved < MAX_PATCHES and attempts < max_attempts:
        y_mask, x_mask = tissue_coords[random.randint(0, len(tissue_coords) - 1)]
        x = int(x_mask * downsample)
        y = int(y_mask * downsample)
        dx = random.randint(0, max(0, int(downsample) - PATCH_SIZE))
        dy = random.randint(0, max(0, int(downsample) - PATCH_SIZE))
        xx = x + dx
        yy = y + dy

        if xx + PATCH_SIZE > slide.dimensions[0] or yy + PATCH_SIZE > slide.dimensions[1]:
            attempts += 1
            continue

        patch = slide.read_region((xx, yy), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        is_valid, invalid_info = is_valid_patch(patch)
        if not is_valid:
            # 保存无效patch到debug目录，文件名中带判别原因和参数
            def format_float(val, ndigits=2):
                try:
                    if val == '' or val is None:
                        return ''
                    return f"{float(val):.{ndigits}f}"
                except Exception:
                    return str(val)

            debug_patch_name = (
                f"inv_{total_invalid+1}_{invalid_info.get('reason','unknown')}"
                f"_W{format_float(invalid_info.get('white_ratio'))}"
                f"_S{format_float(invalid_info.get('sat_mean'), 1)}"
                f"_H{format_float(invalid_info.get('hue_var'), 1)}"
                f"_G{format_float(invalid_info.get('gray_std'), 1)}.jpg"
            )
            debug_patch_path = debug_patient_dir / debug_patch_name
            patch.save(debug_patch_path, "JPEG")
            total_invalid += 1
            attempts += 1
            continue

        patch_path = patient_patch_dir / f"{total_saved + 1}.jpg"
        patch.save(patch_path, "JPEG")
        total_saved += 1
        if total_saved % 25 == 0:
            tqdm.write(f"{tiff_file}: 已保存patch数={total_saved}")
        attempts += 1

    if attempts >= max_attempts and total_saved < MAX_PATCHES:
        tqdm.write(f"⚠️ Patient {patient_id} 达到最大尝试次数，只保存了 {total_saved} 个patch，有效patch不足。")
    tqdm.write(f"✅ Patient {patient_id} 完成，保存 patch 数量: {total_saved}，无效patch数: {total_invalid}")
