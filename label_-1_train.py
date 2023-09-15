import os
import cv2

# ------------------对eiseg标注后的msak进行类别转化，用于训练-------------------

def modify_grayscale_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    modified_image = image.copy()
                    modified_image[modified_image == 0] = 255
                    modified_image[modified_image == 26] = 255
                    modified_image[modified_image < 30]-= 1
                    cv2.imwrite(file_path, modified_image)
# 调用函数修改文件夹中所有PNG灰度图像的像素值，并替换原始图像文件
folder_path = "/home/data/datasets/Map_V11_DB10/manual_annotation/self-chair_manual_annotation/label"
modify_grayscale_images(folder_path)

# import os
# import cv2

# def modify_grayscale_images(folder_path):
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(".png"):
#                 file_path = os.path.join(root, file)
#                 image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#                 if image is not None:
#                     modified_image = image.copy()
#                     modified_image[modified_image == 255] = 0
#                     # modified_image[modified_image == 26] = 255
#                     # modified_image[modified_image < 30]-= 1
#                     cv2.imwrite(file_path, modified_image)
# # 调用函数修改文件夹中所有PNG灰度图像的像素值，并替换原始图像文件
# # folder_path = "/home/data/yang_file/label_utils/manual_annotation/baidu_manual_annotation"
# # modify_grayscale_images(folder_path)