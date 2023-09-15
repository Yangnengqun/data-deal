import os
import cv2
import numpy as np


# ----------------------------查看PNG有哪些像素点------------------------------
def modify_grayscale_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                print(file_path)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    modified_image = image.copy()
                    for i in range(256):
                        
                        result = cv2.inRange(modified_image, i, i)

# 检查结果
                        if np.sum(result) > 0:
                            print("      ",i)
# 调用函数修改文件夹中所有PNG灰度图像的像素值，并替换原始图像文件
folder_path = "/home/data/datasets/Map_V11_DB10/voc/annotations"
modify_grayscale_images(folder_path)