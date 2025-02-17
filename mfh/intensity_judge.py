import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_image_histograms(folder_path):
    all_histograms = np.zeros((256,), dtype=np.float64)
    results = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # 将RGB图像转换为灰度图像，以便分析亮度分布
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算灰度图像的直方图
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        # 累加每张图像的直方图
        all_histograms += hist.flatten()

        # 进行形状判断
        shape_grade = judge_histogram_shape(hist)

        # 进行动态范围评估
        dynamic_range_grade = evaluate_dynamic_range(gray_image)

        results.append([filename, shape_grade, dynamic_range_grade])

    # 绘制所有图像融合在一起的直方图
    plt.plot(all_histograms)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of All Images')
    plt.savefig('all_images_histogram.png')
    plt.close()

    # 将结果转换为合适的数据类型以便保存到CSV文件
    all_histograms = all_histograms.astype(np.int32)
    overall_shape_grade = overall_judge_histogram_shape(all_histograms)
    overall_dynamic_range_grade = overall_evaluate_dynamic_range(all_histograms)

    results_df = pd.DataFrame(results, columns=['Filename', 'Histogram Shape Grade', 'Dynamic Range Grade'])

    # 判断文件是否存在，如果不存在则创建并写入表头和数据
    file_path = 'image_analysis_results.csv'
    if not os.path.exists(file_path):
        results_df.to_csv(file_path, index=False, mode='w', encoding='utf-8')
    else:
        results_df.to_csv(file_path, index=False, mode='a', encoding='utf-8')

    with open(file_path, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"Overall Histogram Shape Grade: {overall_shape_grade}, Overall Dynamic Range Grade: {overall_dynamic_range_grade}\n" + content)


def judge_histogram_shape(hist):
    # 获取直方图的统计信息
    total_pixels = np.sum(hist)
    left_side_pixels = np.sum(hist[:128])
    right_side_pixels = np.sum(hist[128:])

    if left_side_pixels / total_pixels > 0.8:
        return "暗（等级：高）"
    elif right_side_pixels / total_pixels > 0.8:
        return "亮（等级：高）"
    elif left_side_pixels / total_pixels > 0.6:
        return "偏暗（等级：中）"
    elif right_side_pixels / total_pixels > 0.6:
        return "偏亮（等级：中）"
    else:
        return "正常（等级：低）"


def evaluate_dynamic_range(gray_image):
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_image)
    dynamic_range = max_val - min_val

    if dynamic_range < 50:
        return "动态范围小（等级：高）"
    elif dynamic_range < 100:
        import pandas as pd


def analyze_image_histograms(folder_path):
    all_histograms = np.zeros((256,), dtype=np.float64)
    results = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # 将RGB图像转换为灰度图像，以便分析亮度分布
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算灰度图像的直方图
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        # 累加每张图像的直方图
        all_histograms += hist.flatten()

        # 进行形状判断
        shape_grade = judge_histogram_shape(hist)

        # 进行动态范围评估
        dynamic_range_grade = evaluate_dynamic_range(gray_image)

        results.append([filename, shape_grade, dynamic_range_grade])

    # 绘制所有图像融合在一起的直方图
    plt.plot(all_histograms)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of All Images')
    plt.savefig('intensity_histogram.png')
    plt.close()

    # 将结果转换为合适的数据类型以便保存到CSV文件
    all_histograms = all_histograms.astype(np.int32)
    overall_shape_grade = overall_judge_histogram_shape(all_histograms)
    overall_dynamic_range_grade = overall_evaluate_dynamic_range(all_histograms)

    results_df = pd.DataFrame(results, columns=['Filename', 'Histogram Shape Grade', 'Dynamic Range Grade'])

    # 判断文件是否存在，如果不存在则创建并写入表头和数据
    file_path = 'intensity.csv'
    if not os.path.exists(file_path):
        results_df.to_csv(file_path, index=False, mode='w', encoding='utf-8')
    else:
        results_df.to_csv(file_path, index=False, mode='a', encoding='utf-8')

    with open(file_path, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"Overall Histogram Shape Grade: {overall_shape_grade}, Overall Dynamic Range Grade: {overall_dynamic_range_grade}\n" + content)


def judge_histogram_shape(hist):
    # 获取直方图的统计信息
    total_pixels = np.sum(hist)
    left_side_pixels = np.sum(hist[:128])
    right_side_pixels = np.sum(hist[128:])

    if left_side_pixels / total_pixels > 0.8:
        return "暗（等级：高）"
    elif right_side_pixels / total_pixels > 0.8:
        return "亮（等级：高）"
    elif left_side_pixels / total_pixels > 0.6:
        return "偏暗（等级：中）"
    elif right_side_pixels / total_pixels > 0.6:
        return "偏亮（等级：中）"
    else:
        return "正常（等级：低）"


def evaluate_dynamic_range(gray_image):
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_image)
    dynamic_range = max_val - min_val

    if dynamic_range < 50:
        return "动态范围小（等级：高）"
    elif dynamic_range < 100:
        return "动态范围较小（等级：中）"
    else:
        return "动态范围正常（等级：低）"


def overall_judge_histogram_shape(all_histograms):
    total_pixels = np.sum(all_histograms)
    left_side_pixels = np.sum(all_histograms[:128])
    right_side_pixels = np.sum(all_histograms[128:])

    if left_side_pixels / total_pixels > 0.8:
        return "暗（等级：高）"
    elif right_side_pixels / total_pixels > 0.8:
        return "亮（等级：高）"
    elif left_side_pixels / total_pixels > 0.6:
        return "偏暗（等级：中）"
    elif right_side_pixels / total_pixels > 0.6:
        return "偏亮（等级：中）"
    else:
        return "正常（等级：低）"


def overall_evaluate_dynamic_range(all_histograms):
    min_val = np.min(all_histograms)
    max_val = np.max(all_histograms)
    dynamic_range = max_val - min_val

    if dynamic_range < 50:
        return "动态范围小（等级：高）"
    elif dynamic_range < 100:
        return "动态范围较小（等级：中）"
    else:
        return "动态范围正常（等级：低）"


if __name__ == "__main__":
    folder_path = "../datanbi/nbitrain/images"
    analyze_image_histograms(folder_path)
