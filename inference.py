# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -i./ --box "[95,255,190,350]"

"""

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    # x0, y0 = box[0], 0, 0
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
# 修改此处，将 -i 参数改为可以接受文件夹路径，添加 --recursive 参数用于指定是否递归遍历子文件夹
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    # default="assets/test1/img_demo.png",
    # default="assets/test1",
    default="datanbi/nbitest/images_test",
    help="path to the data folder or file. If a folder is provided "
         "and --recursive is set, all images in the folder "
         "(and subfolders if recursive) will be processed.",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    # default="assets/segs",
    default="datanbi/nbitest/segs_test",
    help="path to the_segmentation_folder",
)
parser.add_argument(
    "--box",
    type=str,
    # default='[95, 255, 190, 350]',
    # default='[0, 0, 1249, 1079]',
    default='[0, 0, 1279, 1079]',
    # default='[0, 0, 1023, 1023]',
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    # default="work_dir/MedSAM/medsam_vit_b.pth",
    default="work_dir/MedSAM3/medsam_model_best_convert.pth",
    help="path to the trained model",
)
# 添加新的参数，用于指定是否递归遍历子文件夹
parser.add_argument(
    "--recursive",
    action="store_true",
    # default=False,
    default=True,
    help="Recursively traverse subfolders if set to True when the data path is a folder."
)
args = parser.parse_args()

device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()


# 定义函数用于获取指定路径下的所有图像文件路径
def get_image_paths(folder_path, recursive=False):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']  # 常见的图像文件扩展名
    image_paths = []
    if recursive:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            if any(file.endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(folder_path, file))
    return image_paths


# 根据传入的参数决定是处理单张图片还是文件夹下的所有图片
if os.path.isfile(args.data_path):
    image_paths = [args.data_path]
else:
    image_paths = get_image_paths(args.data_path, args.recursive)


for image_path in image_paths:
    img_np = io.imread(image_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    # %% image preprocessing
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]])
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)

    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
    io.imsave(
        join(args.seg_path, "seg_" + os.path.basename(image_path)),
        (medsam_seg * 255).astype(np.uint8),
        check_contrast=False,
    )

    # # %% visualize results
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(img_3c)
    # show_box(box_np[0], ax[0])
    # ax[0].set_title("Input Image and Bounding Box")
    # ax[1].imshow(img_3c)
    # show_mask(medsam_seg, ax[1])
    # show_box(box_np[0], ax[1])
    # ax[1].set_title("MedSAM Segmentation")
    #
    # # 添加以下代码，保存可视化后的图像
    # visual_result_path = join(args.seg_path, "visual_" + os.path.basename(image_path))
    # plt.savefig(visual_result_path)
    #
    # plt.show()
