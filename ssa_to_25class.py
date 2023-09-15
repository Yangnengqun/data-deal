import os
import json
import cv2 as cv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shutil

# def read_dirs(f_path, jsons=[]):
#     # 获取f_path路径下的所有文件及文件夹
#     paths = os.listdir(f_path)
#     # 判断
#     for f_name in paths:
#         com_path = os.path.join(f_path, f_name)
#         if os.path.isdir(com_path):  # 如果是一个文件夹 
#             read_dirs(com_path, jsons)    # 递归调用
#         if os.path.isfile(com_path):    # 如果是一个文件
#             try:
#                 suffix = com_path.split(".")[-1]  # suffix=后缀（获取文件的后缀）
#             except Exception as e:
#                 continue    # 对于没有后缀的文件省略跳过
#             try:
#                 if suffix == "jpg" or suffix == "png" or suffix == "PNG" or suffix == "JPG" or suffix == "jpeg" or suffix == "webp" or suffix == "json":   
#                     jsons.append(com_path)
#                 else:
#                     continue
#             except Exception as e:
#                 print(e)
#                 continue

# ssa所有的数值加1  再检查的时候key-1
ade20k_class_remap = {1:8, 2:8, 3:255, 4:1, 5:10, 6:8, 7:1, 8:2, 9:17,
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

def ssa_to_png(json_root,png_dir):
    if not os.path.exists(png_dir):
        os.mkdir(png_dir)
    jsons = os.listdir(json_root)
    for json_name in jsons:
        if json_name.endswith(".json"):
            json_path = os.path.join(json_root, json_name)
            with open(json_path, 'r') as j:
                json_content = json.loads(j.read())
            init_flag = True
            
            for id_str, mask in json_content['semantic_mask'].items():
                mask_ = maskUtils.decode(mask)
                h, w = mask_.shape
                if init_flag:
                    image = np.full((h, w), 255, dtype=np.uint8)
                    init_flag = False
                id = int(id_str)+1
                id_real = ade20k_class_remap[id]
                print(id_real)
                image[mask_==1] = id_real
            sv_path = os.path.join(png_dir, json_name[:-5]+".png")
            
            cv.imwrite(sv_path, image)

json_root = "/home/data/datasets/Map_V11_DB10/voc/ssa_out"
png_dir = '/home/data/datasets/Map_V11_DB10/voc/ssa_mask'           
ssa_to_png(json_root,png_dir)
# json_root = "/home/data/yang_file/data_deal/problem_image/ADE20K/ssa_out/ADE_train_00000259_semantic.json"
# png_dir = '/home/data/yang_file/data_deal/problem_image/ADE20K/pid_mask'           
# if not os.path.exists(png_dir):
#     os.mkdir(png_dir)
# with open(json_root, 'r') as j:
#     json_content = json.loads(j.read())
#     init_flag = True
            
#             # for anno in  json_content['annotations']:
#             #     mask = anno['segmentation']
#             #     mask_ = maskUtils.decode(mask)
#             #     h, w = mask_.shape
#             #     if init_flag:
#             #         seg_mask = torch.zeros((1, 1, h, w))
#             #         init_flag = False
#             #     mask_ = torch.from_numpy(mask_).unsqueeze(0).unsqueeze(0)
#             #     seg_mask[mask_] = int(id_str)
#             # seg_logit = torch.zeros((1, num_classes, h, w))
#             # seg_logit.scatter_(1, seg_mask.long(), 1)
#             # seg_logit = seg_logit.float()
#             # seg_pred = F.softmax(seg_logit, dim=1).argmax(dim=1).squeeze(0).numpy()
#             # sv_path = os.path.join(sv_root, json_name[:-5]+".png")

#             # cv.imwrite(sv_path, seg_pred)
            
#     for id_str, mask in json_content['semantic_mask'].items():
#         mask_ = maskUtils.decode(mask)
#         h, w = mask_.shape
#         if init_flag:
#             image = np.full((h, w), 255, dtype=np.uint8)
#             init_flag = False
#         id = int(id_str)+1
#         print(f"id={id}")
#         id_real = ade20k_class_remap[id]
#         print(f"id_real = {id_real}")
#         image[mask_==1] = id_real
#     # seg_logit = torch.zeros((1, 256, h, w))
#     # seg_logit.scatter_(1, seg_mask.long(), 1)
#     # seg_logit = seg_logit.float()
#     # seg_pred = F.softmax(seg_logit, dim=1).argmax(dim=1).squeeze(0).numpy()
#     # seg_pred = seg_mask.squeeze(0).numpy()
#     sv_path = os.path.join(png_dir, 'aaa.png')
    
#     cv.imwrite(sv_path, image)