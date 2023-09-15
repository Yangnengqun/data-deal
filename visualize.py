# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # 读取灰度图像
# dir = "/home/data/yang_file/label_utils/manual_annotation/baidu_manual_annotation/baby_chair_baidu_1/label/1_2.png"
# gray_image = Image.open(dir).convert('L')

# gray_array = np.array(gray_image)
# print(gray_array)
# # 自定义颜色映射
# def custom_colormap(gray_value):
#     # 定义颜色映射
#     color_map_25= [
#                 [220,20,60],
#                 [80, 50, 50],
#                 [ 204, 5, 255],
#                 [150, 5, 61],
#                 [8, 255, 51],
#                 [255, 6, 82],
#                 [204, 70, 3],
#                 [20, 255, 0],
#                 [120, 120, 120],
#                 [255, 0, 122],
#                 [204, 255, 4],
#                 [235, 12, 255],
#                 [224, 5, 255],
#                 [11, 102, 255],
#                 [0, 255, 163],
#                 [0, 255, 194],
#                 [255, 6, 51],
#                 [230, 230, 230],
#                 [255, 51, 7],
#                 [255, 184, 6],
#                 [255, 224, 0],
#                 [0, 255, 20],
#                 [173, 0, 255],
#                 [0, 255, 204],
#                 [0, 0, 255],
#                 [255,255,255]]
#     # print(len(color_map_25))
    
#     # 计算灰度值在颜色映射中的位置
    
#     # 返回对应的颜色
#     return color_map_25[gray_value]

# # 创建彩色图像
# colored_array = np.zeros((gray_array.shape[0], gray_array.shape[1], 3), dtype=np.uint8)
# gray_array[gray_array==255]=25
# # gray_array=gray_array-1

# for i in range(gray_array.shape[0]):
#     for j in range(gray_array.shape[1]):
#         colored_array[i, j] = custom_colormap(gray_array[i, j])

# # 显示彩色图像
# plt.imshow(colored_array)
# plt.axis('off')  # 关闭坐标轴
# plt.show()


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# def custom_colormap(gray_value):
#     # 自定义颜色映射表
#     color_map_25 = [
#         [220, 20, 60], [80, 50, 50], [204, 5, 255], [150, 5, 61], [8, 255, 51],
#         [255, 6, 82], [204, 70, 3], [20, 255, 0], [120, 120, 120], [255, 0, 122],
#         [204, 255, 4], [235, 12, 255], [224, 5, 255], [11, 102, 255], [0, 255, 163],
#         [0, 255, 194], [255, 6, 51], [230, 230, 230], [255, 51, 7], [255, 184, 6],
#         [255, 224, 0], [0, 255, 20], [173, 0, 255], [0, 255, 204], [0, 0, 255],
#         [255, 255, 255]
#     ]
#     return color_map_25[gray_value]

# def show_images_in_folder(folder_path):
#     pid_mask_root = os.path.join(folder_path,"pid_mask")
#     anno_root = os.path.join(folder_path,"annotations")
#     # anno_root = os.path.join(folder_path,"output_sam_oneformer_mask")
#     image_root = os.path.join(folder_path,"images")
#     for filename in os.listdir(pid_mask_root):
#         pid_mask_path = os.path.join(pid_mask_root, filename)
#         anno_path = os.path.join(anno_root, filename)
#         image_path = os.path.join(image_root, filename.replace(".png", ".jpg"))
#         # image_path = os.path.join(image_root, filename.replace(".png_semantic.png", ".png"))
#         if os.path.isfile(pid_mask_path ) and filename.endswith('.png') and os.path.isfile(anno_path ) and os.path.isfile(image_path ) :
#             pid_image = Image.open(pid_mask_path )
#             anno_image = Image.open(anno_path )
#             ori_image = Image.open(image_path )
#             pid_gray_array = np.array(pid_image.convert('L'))
#             anno_gray_array = np.array(anno_image.convert('L'))
#             ori_image_array = np.array(ori_image)

#             # print(gray_array)
            
#             # colored_array = np.zeros((gray_array.shape[0], gray_array.shape[1], 3), dtype=np.uint8)
#             # gray_array[gray_array == 255] = 25
#             # for i in range(gray_array.shape[0]):
#             #     for j in range(gray_array.shape[1]):
#             #         colored_array[i, j] = custom_colormap(gray_array[i, j])
#             plt.figure(0)
#             plt.subplot(131)
#             plt.imshow(pid_gray_array, cmap ='gray')
#             plt.subplot(132)
#             plt.imshow(anno_gray_array,cmap ='gray')
#             plt.subplot(133)
#             plt.imshow(ori_image_array)
#             plt.axis('off')
#             plt.show()

# # 调用函数显示文件夹中所有的图片
# folder_path = '/home/data/yang_file/label_utils/ADE20K'
# show_images_in_folder(folder_path)


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def custom_colormap(gray_value):
    # 自定义颜色映射表
    color_map_25 = [
        [220, 20, 60], [80, 50, 50], [204, 5, 255], [150, 5, 61], [8, 255, 51],
        [255, 6, 82], [204, 70, 3], [20, 255, 0], [120, 120, 120], [255, 0, 122],
        [204, 255, 4], [235, 12, 255], [224, 5, 255], [11, 102, 255], [0, 255, 163],
        [0, 255, 194], [255, 6, 51], [230, 230, 230], [255, 51, 7], [255, 184, 6],
        [255, 224, 0], [0, 255, 20], [173, 0, 255], [0, 255, 204], [0, 0, 255],
        [255, 255, 255]
    ]
    return color_map_25[gray_value]

def show_images_in_folder(folder_path):
    color_root = os.path.join(folder_path,"color_class")
    image_root = os.path.join(folder_path,"images")
    i = 0
    lis_1 = os.listdir(image_root)
    lis=sorted(lis_1)
    maxlis = len(lis)
    print(maxlis)
    for file_num in range(i,maxlis):
    # for filename in os.listdir(image_root):
        filename = lis[file_num]
        image_path = os.path.join(image_root, filename)
        # image_path = os.path.join(image_root, filename.replace(".png", ".jpg"))
        color_path = os.path.join(color_root, filename)
        if os.path.isfile(image_path ) and filename.endswith('.jpg')and os.path.isfile(color_path ) :
            color_image = Image.open(color_path )
            ori_image = Image.open(image_path )
            color_array = np.array(color_image)
            ori_image_array = np.array(ori_image)
            i = i+1
            print(i)
            print(filename)

            # print(gray_array)
            
            # colored_array = np.zeros((pid_gray_array.shape[0], pid_gray_array.shape[1], 3), dtype=np.uint8)
            # pid_gray_array[pid_gray_array == 255] = 25
            # for i in range(pid_gray_array.shape[0]):
            #     for j in range(pid_gray_array.shape[1]):
            #         colored_array[i, j] = custom_colormap(pid_gray_array[i, j])
            
            # 显示灰度图像、彩色图像和原始图像
            plt.figure(figsize=(12, 8))  # 设置图像的大小，单位是英寸
            plt.subplot(121)
            plt.imshow(color_array)  # 手动设置宽高比为1:1
            plt.title('Colored Image')
            plt.subplot(122)
            plt.imshow(ori_image_array)  # 手动设置宽高比为1:1
            plt.title(filename)
            plt.axis('off')
            plt.tight_layout()  # 自动调整子图布局，防止重叠
            plt.show()


# 调用函数显示文件夹中所有的图片
folder_path = '/home/data/yang_file/data_deal/problem_image/merge_image_indoor_cvpr2009'
show_images_in_folder(folder_path)