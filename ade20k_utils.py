# -*-coding:utf-8-*-
import os
import shutil
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from multiprocessing import Pool

ade20k_class_remap = {0:0, 1:8, 2:8, 3:255, 4:1, 5:10, 6:8, 7:1, 8:2, 9:17,
     10:255, 11:12, 12:1, 13:3, 14:1, 15:4, 16:5, 17:255, 18:10, 19:18, 20:6, 21:255, 22:255, 23:16,
     24:13, 25:12, 26:255, 27:255, 28:255, 29:1, 30:255, 31:6, 32:6, 33:19, 34:5, 35:255, 36:12,
     37:0, 38:255, 39:19, 40:255, 41:255, 42:21, 43:8, 44:16, 45:12, 46:11, 47:1, 48:255, 49:255,
     50:255, 51:7, 52:255, 53:1, 54:20, 55:1, 56:12, 57:5, 58:255, 59:17, 60:20, 61:255, 62:255, 63:12,
     64:255, 65:5, 66:255, 67:10, 68:0, 69:255, 70:6, 71:255, 72:255, 73:10, 74:5, 75:15, 76:6, 77:255, 78:11,
     79:255, 80:255, 81:255, 82:255, 83:0, 84:255, 85:255, 86:0, 87:255, 88:0, 89:255, 90:15, 91:255, 92:1,
     93:255, 94:255, 95:1, 96:19, 97:14, 98:255, 99:23, 100:12, 101:16, 102:255, 103:255, 104:255, 105:255,
     106:14, 107:255, 108:255, 109:255,110:255, 111:6, 112:255, 113:0, 114:255, 115:255, 116:0, 117:255,
     118:255, 119:255, 120:0, 121:255, 122:20, 123:255, 124:16, 125:255, 126:23, 127:9, 128:255, 129:255, 130:255,
     131:15, 132:255, 133:0, 134:255, 135:255, 136:23, 137:255, 138:255, 139:22, 140:255, 141:255, 142:15, 143:255,
     144:15, 145:16, 146:255, 147:255, 148:255, 149:255, 150:255 }
    
    
def remap_ade20k():
    root="/media/robot/nvme2T/18_seg_datesets/mseg_dataset/Map_V10/auto_label_datasets_merge1"
    image_names = os.listdir(os.path.join(root, 'images'))
    sv_root = os.path.join(root, "pid_mask")
    if not os.path.exists(sv_root):
        os.makedirs(sv_root, mode=0o777, exist_ok=False)
    for im_name in image_names:
        # image_path = os.path.join(image_root, "origin", im_name)
        mask_name = im_name + "_semantic.png"
        mask_path = os.path.join(root, "output_sam_oneformer_mask", mask_name)
        mask_mat_np = np.array(Image.open(mask_path)) + 1
        mask_cp = np.zeros(mask_mat_np.shape)
        
        for idx in ade20k_class_remap:
            label_idx = int(ade20k_class_remap[idx])
            coords = np.where(mask_mat_np == idx)
            if len(coords[0]) <= 0:
                continue
            else:
                coord_1 = coords[0]
                coord_2 = coords[1]
                mask_cp[[coord_1, coord_2]] = label_idx

        sv_path = os.path.join(sv_root, mask_name)
        cv.imwrite(sv_path, mask_cp)


def plot_remask_show(image, label25, label):
    plt.figure(0)
    plt.subplot(131)
    plt.title("origin")
    plt.imshow(image)
    plt.subplot(132)
    plt.title("label25")
    plt.imshow(label25)
    plt.subplot(133)
    plt.imshow(label)
    plt.title("ade20k")
    plt.show()


def show_item():
    root = "/media/robot/nvme2T/18_seg_datesets/mseg_dataset/Map_V10/auto_label_datasets_merge1"
    
    image_names = os.listdir(os.path.join(root, 'images'))
    
    for im_name in image_names:
        mask_name = im_name + "_semantic.png"
        mask_path = os.path.join(root, "output_sam_oneformer_mask", mask_name)
        remask_path = os.path.join(root, "remask25", mask_name)
        origin_path = os.path.join(root, "images", im_name)
        
        label = cv.imread(mask_path, 0)
        image = cv.imread(origin_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        re_label = cv.imread(remask_path, 0)
        
        plot_remask_show(image, re_label, label)
        
        aa = 0


def show_item_id(id_class):
    root = "/media/robot/nvme2T/18_seg_datesets/mseg_dataset/Map_V10/auto_label_datasets_merge1"
    image_names = os.listdir(os.path.join(root, 'images'))
    for im_name in image_names:
        mask_name = im_name + "_semantic.png"
        mask_path = os.path.join(root, "output_sam_oneformer_mask", mask_name)
        remask_path = os.path.join(root, "remask25", mask_name)
        origin_path = os.path.join(root, "images", im_name)
        
        label = cv.imread(mask_path, 0) + 1
        image = cv.imread(origin_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        re_label = cv.imread(remask_path, 0)
        
        if id_class in label:
        
            plot_remask_show(image, re_label, label)
        
        aa = 0


def map_ade20k_official():
    label_17_root = "/media/robot/nvme2T/18_seg_datesets/mseg_dataset/Map_V10/ADE20K/annotations"
    need_map_names = os.listdir(label_17_root)
    ori_root = "/media/robot/4T/18_seg_datesets/mseg_dataset/MsegV2/ADE20K/ADEChallengeData2016/annotations/training"
    sv_root = "/media/robot/nvme2T/18_seg_datesets/mseg_dataset/Map_V10/ADE20K/pid_mask"
    
    for name in need_map_names:
        mask_150_path = os.path.join(ori_root, name)
        if not os.path.exists(mask_150_path):
            # print("not exists:", mask_150_path)
            mask_150_path = os.path.join(ori_root.replace("training", "validation"), name)
            if not os.path.exists(mask_150_path):
                print("not exists:", mask_150_path)
                exit()
        # else:
        mask_mat_np = np.array(Image.open(mask_150_path)) + 1
        sv_path = os.path.join(sv_root, name)
        
        if os.path.exists(sv_path):
            continue
        
        mask_cp = np.zeros(mask_mat_np.shape)

        for idx in ade20k_class_remap:
            label_idx = int(ade20k_class_remap[idx])
            coords = np.where(mask_mat_np == idx)
            if len(coords[0]) <= 0:
                continue
            else:
                coord_1 = coords[0]
                coord_2 = coords[1]
                mask_cp[[coord_1, coord_2]] = label_idx
        cv.imwrite(sv_path, mask_cp)

def remap_self_image():
    root="/media/robot/nvme2T/18_seg_datesets/mseg_dataset/Map_V10/self-data"
    image_names = os.listdir(os.path.join(root, 'self-images'))
    sv_root = os.path.join(root, "pid_mask")
    if not os.path.exists(sv_root):
        os.makedirs(sv_root, mode=0o777, exist_ok=False)
    for im_name in image_names:
        # image_path = os.path.join(image_root, "origin", im_name)
        mask_name = im_name + "_semantic.png"
        mask_path = os.path.join(root, "output_sam_oneformer_self_images_mask", mask_name)
        mask_mat_np = np.array(Image.open(mask_path)) + 1
        mask_cp = np.zeros(mask_mat_np.shape)
        
        for idx in ade20k_class_remap:
            label_idx = int(ade20k_class_remap[idx])
            coords = np.where(mask_mat_np == idx)
            if len(coords[0]) <= 0:
                continue
            else:
                coord_1 = coords[0]
                coord_2 = coords[1]
                mask_cp[[coord_1, coord_2]] = label_idx

        sv_path = os.path.join(sv_root, mask_name)
        cv.imwrite(sv_path, mask_cp)
        
if __name__ == "__main__":
    # remap_ade20k()
    map_ade20k_official()
    # remap_self_image()
    # show_item()
    # show_item_id(id_class=139)