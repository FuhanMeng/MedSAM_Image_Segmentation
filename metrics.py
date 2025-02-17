import os
from os.path import join
import torch
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_image_as_tensor(image_path):
    """加载单张图像并转换为 PyTorch张量."""
    image = Image.open(image_path).convert('L')  # 转为灰度图像
    image = np.array(image) / 255.0  # 归一化到 [0, 1]
    return torch.tensor(image).unsqueeze(0).unsqueeze(0).float()  # 添加 batch 和 channel 维度


def get_metrics(pred, mask):
    """计算分割评价指标.accuracy受背景影响大，mae受背景影响小，其余指标不受背景影响"""
    pred = (pred > 0.5).float()
    pred_positives = pred.sum(dim=(2, 3))
    mask_positives = mask.sum(dim=(2, 3))
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred_positives + mask_positives
    dice = (2 * inter) / (union + 1e-6)
    iou = inter / (union - inter + 1e-6)
    acc = (pred == mask).float().mean(dim=(2, 3))
    recall = inter / (mask_positives + 1e-6)
    precision = inter / (pred_positives + 1e-6)
    f2 = (5 * inter) / (4 * mask_positives + pred_positives + 1e-6)
    mae = (torch.abs(pred - mask)).mean(dim=(2, 3))

    return dice, iou, acc, recall, precision, f2, mae


def get_metrics_withoutbg(pred, mask):
    """计算分割评价指标，完全去除背景的影响."""
    pred = (pred > 0.5).float()  # 二值化预测结果
    mask = mask.float()  # 确保mask是浮点类型

    # 只考虑前景区域（mask中值为1的像素）
    foreground_mask = (mask == 1)  # 前景区域的掩码
    pred_foreground = pred[foreground_mask]  # 只取前景区域的预测值
    mask_foreground = mask[foreground_mask]  # 只取前景区域的真值

    # 如果没有前景区域，直接返回0
    if mask_foreground.sum() == 0:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    # 计算指标
    inter = (pred_foreground * mask_foreground).sum()
    pred_positives = pred_foreground.sum()
    mask_positives = mask_foreground.sum()
    union = pred_positives + mask_positives

    dice = (2 * inter) / (union + 1e-6)
    iou = inter / (union - inter + 1e-6)
    acc = (pred_foreground == mask_foreground).float().mean()
    recall = inter / (mask_positives + 1e-6)
    precision = inter / (pred_positives + 1e-6)
    f2 = (5 * inter) / (4 * mask_positives + pred_positives + 1e-6)
    mae = (torch.abs(pred_foreground - mask_foreground)).mean()

    return dice, iou, acc, recall, precision, f2, mae


# 设置文件夹路径
# gt_dir = 'datanbi/nbitest/labels'  # 真值文件夹路径
# seg_dir = 'datanbi/nbitest/segs3'  # 预测结果文件夹路径
# output_csv = 'datanbi/nbitest/metrics3/metrics_results.csv'  # 结果保存路径
gt_dir = 'datanbi/nbitest/images_masks'  # 真值文件夹路径
seg_dir = 'datanbi/nbitest/yuanzhe_pre'  # 预测结果文件夹路径
output_csv = 'datanbi/nbitest/yuanzhe_metrics/metrics_results.csv'  # 结果保存路径

# 获取文件列表
gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.png')])

# 确保文件名匹配
if len(gt_files) != len(seg_files):
    raise ValueError("两个文件夹中的图像数量不匹配！请检查文件内容。")

# 创建存储结果的 DataFrame
metrics_data = {
    'Image': [],
    'Dice': [],
    'IoU': [],
    'Accuracy': [],
    'Recall': [],
    'Precision': [],
    'F2-score': [],
    'MAE': []
}

# 用于计算平均值的临时变量
sum_metrics = {
    'Dice': 0,
    'IoU': 0,
    'Accuracy': 0,
    'Recall': 0,
    'Precision': 0,
    'F2-score': 0,
    'MAE': 0
}

# 遍历所有图像并计算指标
for gt_file, seg_file in tqdm(zip(gt_files, seg_files), total=len(gt_files), desc="Processing images"):
    if gt_file.replace('image_', '') != seg_file.replace('seg_image_', ''):
        raise ValueError(f"文件名不匹配: {gt_file} 和 {seg_file}")

    # 加载图像
    gt_path = join(gt_dir, gt_file)
    seg_path = join(seg_dir, seg_file)

    gt_tensor = load_image_as_tensor(gt_path)
    seg_tensor = load_image_as_tensor(seg_path)

    # 计算指标
    dice, iou, acc, recall, precision, f2, mae = get_metrics(seg_tensor, gt_tensor)

    # 保存单张图像结果
    metrics_data['Image'].append(gt_file)
    metrics_data['Dice'].append(round(dice.item(), 4))
    metrics_data['IoU'].append(round(iou.item(), 4))
    metrics_data['Accuracy'].append(round(acc.item(), 4))
    metrics_data['Recall'].append(round(recall.item(), 4))
    metrics_data['Precision'].append(round(precision.item(), 4))
    metrics_data['F2-score'].append(round(f2.item(), 4))
    metrics_data['MAE'].append(round(mae.item(), 4))

    # 累加到平均值计算中
    sum_metrics['Dice'] += dice.item()
    sum_metrics['IoU'] += iou.item()
    sum_metrics['Accuracy'] += acc.item()
    sum_metrics['Recall'] += recall.item()
    sum_metrics['Precision'] += precision.item()
    sum_metrics['F2-score'] += f2.item()
    sum_metrics['MAE'] += mae.item()

# 计算平均值
num_images = len(gt_files)
avg_metrics = {key: round(value / num_images, 4) for key, value in sum_metrics.items()}

# 插入平均值到 DataFrame 的第一行
metrics_data['Image'].insert(0, 'Average')
metrics_data['Dice'].insert(0, avg_metrics['Dice'])
metrics_data['IoU'].insert(0, avg_metrics['IoU'])
metrics_data['Accuracy'].insert(0, avg_metrics['Accuracy'])
metrics_data['Recall'].insert(0, avg_metrics['Recall'])
metrics_data['Precision'].insert(0, avg_metrics['Precision'])
metrics_data['F2-score'].insert(0, avg_metrics['F2-score'])
metrics_data['MAE'].insert(0, avg_metrics['MAE'])

# 保存结果到 CSV 文件
df = pd.DataFrame(metrics_data)
df.to_csv(output_csv, index=False)

print(f"指标计算完成，结果已保存到 {output_csv}")
