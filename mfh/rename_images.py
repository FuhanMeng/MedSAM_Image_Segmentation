import os


def rename_images_and_labels(images_folder, labels_folder, prefix="image"):
    """
    将两个文件夹中的图片按顺序重命名为统一格式（如 image_0001），
    确保两个文件夹中对应的文件名保持一致。

    参数：
    images_folder: 存放原始图片的文件夹路径。
    labels_folder: 存放对应标注图像的文件夹路径。
    prefix: 文件名前缀，默认是'image'。
    """
    # 获取两个文件夹中的文件名，并按字母顺序排序
    image_files = sorted(os.listdir(images_folder))
    label_files = sorted(os.listdir(labels_folder))

    # 检查文件数量是否一致
    if len(image_files) != len(label_files):
        raise ValueError("两个文件夹中的文件数量不一致，请检查！")

    # 初始化计数器
    count = 1

    for image_file, label_file in zip(image_files, label_files):
        # 获取原始图片和标注图片的完整路径
        old_image_path = os.path.join(images_folder, image_file)
        old_label_path = os.path.join(labels_folder, label_file)

        # 确保它们都是文件
        if os.path.isfile(old_image_path) and os.path.isfile(old_label_path):
            # 提取文件扩展名
            _, image_ext = os.path.splitext(image_file)
            _, label_ext = os.path.splitext(label_file)

            # 构造新的文件名
            new_name = f"{prefix}_{count:04d}"
            new_image_name = f"{new_name}{image_ext}"
            new_label_name = f"{new_name}{label_ext}"

            # 构造新的完整路径
            new_image_path = os.path.join(images_folder, new_image_name)
            new_label_path = os.path.join(labels_folder, new_label_name)

            # 重命名文件
            os.rename(old_image_path, new_image_path)
            os.rename(old_label_path, new_label_path)
            print(f"Renamed: {image_file} -> {new_image_name}, {label_file} -> {new_label_name}")

            # 更新计数器
            count += 1


# 示例用法
images_folder = "datanbi/nbitrain/images"  # 替换为你的images文件夹路径
labels_folder = "datanbi/nbitrain/labels"  # 替换为你的labels文件夹路径
rename_images_and_labels(images_folder, labels_folder)
