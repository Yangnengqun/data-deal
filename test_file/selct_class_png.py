import os
import cv2
import numpy as np
import shutil

# -----------------------选取含有指定类别（0-24）的灰度图片，用于可视化---------------------
def selct_image_and_png(png_root, image_root,gt_mask_root,class_num, out_root):
    png_list = os.listdir(png_root)
    class_png_root = os.path.join(out_root,"test_png_dir")
    class_image_root = os.path.join(out_root,"image_dir")
    class_gt_mask_root = os.path.join(out_root,"gt_mask")
    if not os.path.exists(class_num):
        os.mkdir(out_root)
    if not os.path.exists(class_png_root):
        os.mkdir(class_png_root)
    # if not os.path.exists(class_image_root):
    #     os.mkdir(class_image_root)
    # if not os.path.exists(class_gt_mask_root):
    #     os.mkdir(class_gt_mask_root)
  
    for png_name in png_list:
        png_dir = os.path.join(png_root,png_name)
        gt_mask_dir = os.path.join(gt_mask_root,png_name)
        png = cv2.imread(png_dir, cv2.IMREAD_GRAYSCALE)
        if np.any(png==class_num):
            class_png_dir = os.path.join(class_png_root,png_name)
            class_gt_mask_dir = os.path.join(class_gt_mask_root,png_name)
            jpg_name = png_name.replace(".png",".jpg")
            image_dir = os.path.join(image_root,jpg_name)
            if os.path.exists(image_dir):
                class_image_dir = os.path.join(class_image_root,jpg_name)
                shutil.copy(png_dir,class_png_dir)
                # shutil.copy(gt_mask_dir,class_gt_mask_dir)
                # shutil.copy(image_dir,class_image_dir)
            else:
                image_dir = os.path.join(image_root,png_name)
                class_image_dir = os.path.join(class_image_root,png_name)
                if os.path.exists(image_dir):
                    shutil.copy(png_dir,class_png_dir)
                    # shutil.copy(gt_mask_dir,class_gt_mask_dir)
                    # shutil.copy(image_dir,class_image_dir)
                else:
                    print(png_name)



if __name__ == "__main__":
    png_root = "/home/data/yang_file/data_deal/test_file/test_all/test_results"
    image_root = "/home/data/yang_file/data_deal/test_file/test_all/images"
    gt_mask_root = "/home/data/yang_file/data_deal/test_file/test_all/gt_mask"
    class_num = 22
    out_root = f"/home/data/yang_file/data_deal/test_file/select_class_{class_num}"
    selct_image_and_png(png_root, image_root,gt_mask_root,class_num,out_root)
    