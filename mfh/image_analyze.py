# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
#
# # 计算息肉和背景像素比例
# def calculate_pixel_proportions(dataset_path):
#     polyp_pixels_total = 0
#     background_pixels_total = 0
#
#     # 遍历数据集中的所有图像
#     for filename in os.listdir(dataset_path):
#         image_path = os.path.join(dataset_path, filename)
#         image = cv2.imread(image_path, 0)  # 以灰度模式读取图像
#
#         polyp_pixels = np.sum(image == 255)  # 统计白色像素（息肉）数量
#         background_pixels = np.sum(image == 0)  # 统计黑色像素（背景）数量
#
#         polyp_pixels_total += polyp_pixels
#         background_pixels_total += background_pixels
#
#     total_pixels = polyp_pixels_total + background_pixels_total
#     polyp_pixel_ratio = polyp_pixels_total / total_pixels
#     background_pixel_ratio = background_pixels_total / total_pixels
#
#     return polyp_pixel_ratio, background_pixel_ratio
#
#
# # 分析息肉大小分布并绘制直方图
# def analyze_polyp_size_distribution(dataset_path):
#     polyp_sizes = []
#
#     # 遍历数据集中的所有图像
#     for filename in os.listdir(dataset_path):
#         image_path = os.path.join(dataset_path, filename)
#         image = cv2.imread(image_path, 0)  # 以灰度模式读取图像
#
#         contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             polyp_area = cv2.contourArea(contour)
#             polyp_sizes.append(polyp_area)
#
#     plt.hist(polyp_sizes, bins=20)  # 绘制直方图，可根据需要调整 bins 的数量
#     plt.xlabel('Polyp Size (Pixels)')
#     plt.ylabel('Frequency')
#     plt.title('Polyp Size Distribution')
#     current_dir = os.path.dirname(__file__)
#     save_path = os.path.join(current_dir, "Polyp_size_dist.png")
#     plt.savefig(save_path)
#     plt.show()
#
#
# if __name__ == "__main__":
#     dataset_path = "../datanbi/nbitrain/labels"  # 修改为实际的数据集路径
#
#     # 计算像素比例
#     polyp_ratio, background_ratio = calculate_pixel_proportions(dataset_path)
#     print(f"息肉像素占比: {polyp_ratio}")
#     print(f"背景像素占比: {background_ratio}")
#
#     # 分析息肉大小分布并绘制直方图
#     analyze_polyp_size_distribution(dataset_path)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# 计算息肉和背景像素比例
def calculate_pixel_proportions(dataset_path):
    polyp_pixels_total = 0
    background_pixels_total = 0
    image_count = 0

    # 遍历数据集中的所有图像
    for filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path, 0)  # 以灰度模式读取图像
        if image is None:
            continue
        image_count += 1

        polyp_pixels = np.sum(image == 255)  # 统计白色像素（息肉）数量
        background_pixels = np.sum(image == 0)  # 统计黑色像素（背景）数量

        polyp_pixels_total += polyp_pixels
        background_pixels_total += background_pixels

    if image_count == 0:
        return 0, 0

    total_pixels = polyp_pixels_total + background_pixels_total
    polyp_pixel_ratio = polyp_pixels_total / total_pixels
    background_pixel_ratio = background_pixels_total / total_pixels

    return polyp_pixel_ratio, background_pixel_ratio


# 分析息肉大小分布并绘制直方图
def analyze_polyp_size_distribution(dataset_path):
    polyp_sizes = []

    # 遍历数据集中的所有图像
    for filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path, 0)  # 以灰度模式读取图像
        if image is None:
            continue

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            polyp_area = cv2.contourArea(contour)
            polyp_sizes.append(polyp_area)

    small_count = 0
    medium_count = 0
    large_count = 0

    for size in polyp_sizes:
        if size < 50000:
            small_count += 1
        elif size < 200000:
            medium_count += 1
        else:
            large_count += 1

    total_count = len(polyp_sizes)
    small_ratio = small_count / total_count if total_count > 0 else 0
    medium_ratio = medium_count / total_count if total_count > 0 else 0
    large_ratio = large_count / total_count if total_count > 0 else 0

    plt.hist(polyp_sizes, bins=20)  # 绘制直方图，可根据需要调整bins的数量
    plt.xlabel('Polyp Size (Pixels)')
    plt.ylabel('Frequency')
    plt.title('Polyp Size Distribution')

    # 在图上标注各级别占比
    plt.text(plt.xlim()[0] + 1000, plt.ylim()[1] * 0.9, f"Small: {small_ratio:.2f}", color='r')
    plt.text(plt.xlim()[0] + 1000, plt.ylim()[1] * 0.8, f"Medium: {medium_ratio:.2f}", color='r')
    plt.text(plt.xlim()[0] + 1000, plt.ylim()[1] * 0.7, f"Large: {large_ratio:.2f}", color='r')

    current_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_dir, "Polyp_size_dist.png")
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    dataset_path = "../datanbi/nbitrain/labels"  # 修改为实际的数据集路径

    # 计算像素比例
    polyp_ratio, background_ratio = calculate_pixel_proportions(dataset_path)
    print(f"息肉像素占比: {polyp_ratio}")
    print(f"背景像素占比: {background_ratio}")

    # 分析息肉大小分布并绘制直方图
    analyze_polyp_size_distribution(dataset_path)