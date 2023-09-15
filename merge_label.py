
# 将ssa分割的模型和  只含有动物植物的类标签的label结合  
# import os
# import cv2
# def merge_ssa_and_label(ssa_root,label_root,out_root):
#     if not os.path.exists(out_root):
#         os.mkdir(out_root)
#     label_files = os.listdir(label_root)
#     for label_name in label_files:
#         label_path = os.path.join(label_root,label_name)
#         ssa_name = label_name.replace(".png","_semantic.png")
#         ssa_path = os.path.join(ssa_root,ssa_name)
#         if os.path.exists(label_path) and os.path.exists(ssa_path):
#             label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#             ssa = cv2.imread(ssa_path, cv2.IMREAD_GRAYSCALE)
#             mask = label.copy()
#             mask[mask==0] = 1
#             mask[mask>1] = 0
#             out_label = mask*ssa+label
#             out_path = os.path.join(out_root,label_name)
#             cv2.imwrite(out_path, out_label)
#         else:
#             print(label_path)
# ssa_root = "/home/data/datasets/Map_V11_DB10/voc/ssa_mask"
# label_root = "/home/data/datasets/Map_V11_DB10/voc/annotations"
# out_root = "/home/data/datasets/Map_V11_DB10/voc/pid_mask"
# merge_ssa_and_label(ssa_root,label_root,out_root)

# -------------------------将ssa分割的模型和  只含有动物植物的类标签的label结合，并去除分割区域小的类别-----------------------
import os
import cv2
import numpy as  np
def merge_ssa_and_label(images_root,ssa_root,label_root,out_root):
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    images_files = os.listdir(images_root)
    for images_name in images_files:
        label_name = images_name.replace(".jpg",".png")
        label_path = os.path.join(label_root,label_name)
        ssa_name = label_name.replace(".png","_semantic.png")
        ssa_path = os.path.join(ssa_root,ssa_name)
        if os.path.exists(label_path) and os.path.exists(ssa_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            ssa = cv2.imread(ssa_path, cv2.IMREAD_GRAYSCALE)
            mask = label.copy()
            mask[mask==0] = 1
            mask[mask>1] = 0
            out_label = mask*ssa+label
            for i in range(25):

# 统计像素值为target_pixel_value的数量
                count = np.count_nonzero(out_label == i)
                if count<2000:
                    out_label[out_label==i]=255
            
            
            out_path = os.path.join(out_root,label_name)
            cv2.imwrite(out_path, out_label)
        else:
            print(label_path)
images_root = "/home/data/datasets/voc/images"
ssa_root = "/home/data/datasets/voc/ssa_mask"
label_root = "/home/data/datasets/voc/annotations"
out_root = "/home/data/datasets/voc/pid_mask"
merge_ssa_and_label(images_root,ssa_root,label_root,out_root)
            
            
            
            