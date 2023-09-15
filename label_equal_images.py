import os
import cv2
def merge_ssa_and_label(images_root,color_root,pid_mask):
    image_files = os.listdir(images_root)
    for image_name in image_files:
        images_path = os.path.join(images_root,image_name)
        mask_name = image_name.replace(".jpg",".png")
        mask_path = os.path.join(pid_mask,mask_name)
        color_path = os.path.join(color_root,image_name)
        if os.path.exists(images_path) and os.path.exists(mask_path):
            if not os.path.exists(color_path):
                os.remove(images_path)
                os.remove(mask_path)  
        else:
            print(images_path)
images_root = "/home/data/datasets/voc/images"
color_root = "/home/data/datasets/voc/color_class"
pid_mask = "/home/data/datasets/voc/ssa_mask"
merge_ssa_and_label(images_root,color_root,pid_mask)
            