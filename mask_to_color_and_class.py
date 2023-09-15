# import os
# import cv2
# import numpy as np

# # 定义颜色映射表，可以根据需求进行修改
# color_map_25= {
#             0:[220,20,60],
#             1:[80, 50, 50],
#             2:[ 204, 5, 255],
#             3:[150, 5, 61],
#             4:[8, 255, 51],
#             5:[255, 6, 82],
#             6:[204, 70, 3],
#             7:[20, 255, 0],
#             8:[120, 120, 120],
#             9:[255, 0, 122],
#             10:[204, 255, 4],
#             11:[235, 12, 255],
#             12:[224, 5, 255],
#             13:[11, 102, 255],
#             14:[0, 255, 163],
#             15:[0, 255, 194],
#             16:[255, 6, 51],
#             17:[230, 230, 230],
#             18:[255, 51, 7],
#             19:[255, 184, 6],
#             20:[255, 224, 0],
#             21:[0, 255, 20],
#             22:[173, 0, 255],
#             23:[0, 255, 204],
#             24:[0, 0, 255],
#             255:[255,255,255]}


# id2label_25={
#     0: "backgroud",
#     1: "floor",
#     2: "bed",
#     3: "person",
#     4: "door",
#     5: "table",
#     6: "chair",
#     7: "refrigerator",
#     8: "wall",
#     9: "animal",
#     10: "plant",
#     11: "reception",
#     12: "cabinet",
#     13: "sofa",
#     14: "escalator",
#     15: "tv",
#     16: "painting",
#     17: "window",
#     18: "curtain",
#     19: "fences",
#     20: "stair",
#     21: "trunk",
#     22: "trash",
#     23: "vase",
#     24: "babychair"
# }
# def mask_to_color_and_class(gray_folder,output_folder):
# # 定义灰度图所在的文件夹路径和新文件夹路径
#     arr = list(range(25))
#     arr.append(255)

#     # 创建新文件夹
#     os.makedirs(output_folder, exist_ok=True)

#     # 遍历灰度图文件夹中的所有文件
#     for filename in os.listdir(gray_folder):
#         if filename.endswith(".png") or filename.endswith(".jpg"):
#             # 读取灰度图
#             gray_img = cv2.imread(os.path.join(gray_folder, filename), 0)

#             # 创建RGB图像
#             rgb_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.uint8)

#             # 根据灰度值扩展为RGB图像，并记录每个颜色区域的位置
#             color_regions = {}
#             for i in arr:
#                 color_regions[i] = []
#                 rgb_img[gray_img == i] = color_map_25[i]
#                 if i!=255:
#                     contours, _ = cv2.findContours((gray_img == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                     for contour in contours:
#                         M = cv2.moments(contour)
#                         if M["m00"] != 0:
#                             cX = int(M["m10"] / M["m00"])
#                             cY = int(M["m01"] / M["m00"])
#                             color_regions[i].append((cX, cY))

#             # 在每个颜色区域中心打上标签
#             for color, regions in color_regions.items():
#                 for region in regions:
#                     label_position = region
#                     cv2.putText(rgb_img, id2label_25[color], label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#             # 保存处理后的图像到新文件夹
#             output_filename = os.path.join(output_folder, filename)
#             cv2.imwrite(output_filename, rgb_img)

#     print(f"Processed {filename} and saved to {output_filename}")


# if __name__ =="__main__":
#     gray_folder = "/home/data/test_file/input"
#     output_folder = "/home/data/test_file/output"
#     # arr = list(range(25))
#     # arr.append(255)
#     # print(arr)
#     mask_to_color_and_class(gray_folder,output_folder)
id2label_25={
        "0": "backgroud",
        "1": "floor",
        "2": "bed",
        "3": "person",
        "4": "door",
        "5": "table",
        "6": "chair",
        "7": "refrigerator",
        "8": "wall",
        "9": "animal",
        "10": "plant",
        "11": "reception",
        "12": "cabinet",
        "13": "sofa",
        "14": "escalator",
        "15": "tv",
        "16": "painting",
        "17": "window",
        "18": "curtain",
        "19": "fences",
        "20": "stair",
        "21": "trunk",
        "22": "trash",
        "23": "vase",
        "24": "babychair"
    }

color_map_25= {
            0:[220,20,60],
            1:[80, 50, 50],
            2:[ 204, 5, 255],
            3:[150, 5, 61],
            4:[8, 255, 51],
            5:[255, 6, 82],
            6:[204, 70, 3],
            7:[20, 255, 0],
            8:[120, 120, 120],
            9:[255, 0, 122],
            10:[204, 255, 4],
            11:[235, 12, 255],
            12:[224, 5, 255],
            13:[11, 102, 255],
            14:[0, 255, 163],
            15:[0, 255, 194],
            16:[255, 6, 51],
            17:[230, 230, 230],
            18:[255, 51, 7],
            19:[255, 184, 6],
            20:[255, 224, 0],
            21:[0, 255, 20],
            22:[173, 0, 255],
            23:[0, 255, 204],
            24:[0, 0, 255],
            255:[255,255,255]}
import os
import torch
import torch.nn.functional as F
from PIL import Image
import mmcv
from tqdm import tqdm
from mmcv.utils import print_log
from mmdet.core.visualization.image import imshow_det_bboxes
import cv2
import numpy as np
import os

label_list = list(id2label_25.values())


# def mask_to_color_and_class(input_image_path,input_mask_path,output_path):
#     if not os.path.exists(output_path):
#         os.mkdir(output_path)
#     input_images = os.listdir(input_image_path)
#     print(len(input_images))
#     for image in input_images:
#         if image.endswith(".png") or image.endswith(".jpg"):
#             image_path = os.path.join(input_image_path,image)
#             img = cv2.imread(image_path,cv2.IMREAD_COLOR)
#             mask_path = os.path.join(input_mask_path,image+'_semantic.png')
#             if os.path.isfile(image_path) and os.path.isfile(mask_path):
#                 img = cv2.imread(image_path,cv2.IMREAD_COLOR)
#                 gray_mask = cv2.imread(mask_path,0)
#                 semantic_bitmasks = []
#                 semantic_name = []
#                 mask_color = []
#                 for i in range(25):
#                     bitmask = gray_mask==i
#                     bitmask = bitmask.astype(np.uint8)
#                     if not np.all(bitmask == 0):
#                         semantic_name.append(label_list[i])
#                         semantic_bitmasks.append(bitmask)
#                         mask_color.append(color_map_25[i])
#                 if not semantic_bitmasks:
#                     print(image)
#                 else:
#                     imshow_det_bboxes(img,
#                                     bboxes=None,
#                                     labels=np.arange(len(semantic_name)),
#                                     segms=np.stack(semantic_bitmasks),
#                                     class_names=semantic_name,
#                                     mask_color=mask_color,
#                                     font_size=25,
#                                     show=False,
#                                     out_file=os.path.join(output_path, image))



# ------------------将灰度图附上颜色和类别名称---------------------------
def mask_to_color_and_class_no_semantic(input_image_path,input_mask_path,output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    input_images = os.listdir(input_image_path)
    print(len(input_images))
    # count = 0
    for image in input_images:
        if image.endswith(".png") or image.endswith(".jpg"):
            image_path = os.path.join(input_image_path,image)
            img = cv2.imread(image_path,cv2.IMREAD_COLOR)
            mask_name = image.replace('.jpg','.png')
            mask_path = os.path.join(input_mask_path,mask_name)
            if not os.path.exists(mask_path):
                mask_path = os.path.join(input_mask_path,image+"_semantic.png")
            if os.path.isfile(image_path) and os.path.isfile(mask_path):
                img = cv2.imread(image_path,cv2.IMREAD_COLOR)
                gray_mask = cv2.imread(mask_path,0)
                semantic_bitmasks = []
                semantic_name = []
                mask_color = []
                for i in range(25):
                    bitmask = gray_mask==i
                    bitmask = bitmask.astype(np.uint8)
                    if not np.all(bitmask == 0):
                        semantic_name.append(label_list[i])
                        semantic_bitmasks.append(bitmask)
                        mask_color.append(color_map_25[i])
                if not semantic_bitmasks:
                    print(image)
                else:
                    imshow_det_bboxes(img,
                                    bboxes=None,
                                    labels=np.arange(len(semantic_name)),
                                    segms=np.stack(semantic_bitmasks),
                                    class_names=semantic_name,
                                    mask_color=mask_color,
                                    font_size=25,
                                    show=False,
                                    out_file=os.path.join(output_path, image))
                    # count+=1
                    # if count>10:
                    #     break

if __name__ == "__main__":
    input_image_path="/home/data/yang_file/data_deal/test_file/test_all/images"
    input_mask_path="/home/data/yang_file/data_deal/test_file/test_all/test_results"
    output_path="/home/data/yang_file/data_deal/test_file/test_all/test_class_color"
    mask_to_color_and_class_no_semantic(input_image_path,input_mask_path,output_path)