# 检查灰度图和原图是否一一对应，并打印不能对应的image
import os

def check(image_root,mask_root):
    if  not os.path.exists(image_root) or not os.path.exists(mask_root):
        print("no root")
    else:
        image_files = os.listdir(image_root)
        for file in image_files:
            if file.endswith('.jpg') or file.endswith('.png'):
                mask_file = os.path.join(mask_root, file+"_semantic.png")
                if not os.path.exists(mask_file):
                    print(file)
                    
                
if __name__ == "__main__":
    image_root = '/home/data/datasets/Map_V10_DB10_auto_label_datasets_deal/indoorCVPR_09/merge_image_indoor_cvpr2009'
    mask_root = '/home/data/datasets/Map_V10_DB10_auto_label_datasets_deal/indoorCVPR_09/pid_mask'
    check(image_root,mask_root)