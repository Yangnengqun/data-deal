import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --------------------对图片的test输出、mask、置信度、原图 四张图进行可视化------------------------


def show_images_in_folder(select_png_path,folder_path):
    test_color_root = os.path.join(folder_path,"test_class_color")
    image_root = os.path.join(folder_path,"images")
    gt_mask = os.path.join(folder_path,"gt_class_color")
    confidence_root = os.path.join(folder_path,"confidence")
    i = 0
    lis_1 = os.listdir(select_png_path)
    lis=sorted(lis_1)
    maxlis = len(lis)
    print(maxlis)
    for file_num in range(i,maxlis):
    # for filename in os.listdir(image_root):
        filename = lis[file_num]
        confidence_path = os.path.join(confidence_root, filename)
        image_name = filename
        image_path = os.path.join(image_root, image_name)
        if not os.path.isfile(image_path):
            image_name = filename.replace(".png",".jpg")
        # image_path = os.path.join(image_root, filename.replace(".png", ".jpg"))
        image_path = os.path.join(image_root, image_name)
        test_color_path = os.path.join(test_color_root, image_name)
        gt_mask_path = os.path.join(gt_mask, image_name)
        if os.path.isfile(image_path ) and os.path.isfile(gt_mask_path) and os.path.isfile(test_color_path) and  os.path.isfile(confidence_path):
            test_color_image = Image.open(test_color_path )
            ori_image = Image.open(image_path )
            gt_image = Image.open(gt_mask_path )
            confidence = Image.open(confidence_path).convert("L")
            color_array = np.array(test_color_image)
            ori_image_array = np.array(ori_image)
            gt_array = np.array(gt_image)
            confidence_array = np.array(confidence)
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
            plt.figure(figsize=(15, 9))  # 设置图像的大小，单位是英寸
            plt.subplot(221)
            plt.imshow(color_array)  # 手动设置宽高比为1:1
            plt.title('Colored test_Image')
            plt.subplot(222)
            plt.imshow(gt_array)  # 手动设置宽高比为1:1
            plt.title('gt_color')
            plt.subplot(223)
            plt.imshow(confidence_array, cmap='gray')
            plt.title('confidence*100')
            plt.subplot(224)
            plt.imshow(ori_image_array)  # 手动设置宽高比为1:1
            plt.title(filename)
            plt.axis('off')
            
            # plt.subplot(221).set_position([0.1, 0.3, 0.4, 0.4])

            # # 调整第二张图片的显示大小
            # plt.subplot(222).set_position([0.5, 0.3, 0.4, 0.4])

            # # 调整第三张图片的显示大小
            # plt.subplot(223).set_position([0.3, 0.0, 0.4, 0.4])
            # plt.tight_layout()  # 自动调整子图布局，防止重叠
            plt.show()


# 调用函数显示文件夹中所有的图片
select_png_path = "/home/data/yang_file/data_deal/test_file/select_class_21/test_png_dir"
folder_path = '/home/data/yang_file/data_deal/test_file/test_all'
show_images_in_folder(select_png_path, folder_path)