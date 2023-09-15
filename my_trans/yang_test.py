import numpy as np
from skimage.measure import label, regionprops

# def rle_to_mask(rle, img_shape):
#     """
#     将RLE掩码转换为二进制图像掩码
#     :param rle: RLE掩码
#     :param img_shape: 图像形状，例如(Height, Width)
#     :return: 二进制图像掩码
#     """
#     mask = np.zeros(img_shape, dtype=np.uint8)
#     rle = [int(x) for x in rle.split()]
#     starts = rle[0::2]
#     lengths = rle[1::2]
#     for start, length in zip(starts, lengths):
#         start = start - 1  # RLE掩码索引从1开始
#         end = start + length
#         mask[start:end] = 1
#     return mask

# def rle_to_points(rle, img_shape):
#     """
#     将RLE掩码转换为labelme的points格式
#     :param rle: RLE掩码
#     :param img_shape: 图像形状，例如(Height, Width)
#     :return: labelme的points格式
#     """
#     mask = rle_to_mask(rle, img_shape)
#     labeled_mask = label(mask)
#     props = regionprops(labeled_mask)
#     points = []
#     for prop in props:
#         contour = prop.coords.tolist()
#         points.append(contour)
#     return points

# # 示例
# rle = "1 5 10 3 20 4"
# img_shape = (10, 10)
# points = rle_to_points(rle, img_shape)
# print(points)


import pycocotools.mask as mask_util
import numpy as np
import cv2
import json

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
id2label_25_dict={
        0: "backgroud",
        1: "floor",
        2: "bed",
        3: "person",
        4: "door",
        5: "table",
        6: "chair",
        7: "refrigerator",
        8: "wall",
        9: "animal",
        10: "plant",
        11: "reception",
        12: "cabinet",
        13: "sofa",
        14: "escalator",
        15: "tv",
        16: "painting",
        17: "window",
        18: "curtain",
        19: "fences",
        20: "stair",
        21: "trunk",
        22: "trash",
        23: "vase",
        24: "babychair",
        25: "ignore"
    }

# id2label_25_dict={
#         "0": "backgroud",
#         "1": "floor",
#         "2": "bed",
#         "3": "person",
#         "4": "door",
#         "5": "table",
#         "6": "chair",
#         "7": "refrigerator",
#         "8": "wall",
#         "9": "animal",
#         "10": "plant",
#         "11": "reception",
#         "12": "cabinet",
#         "13": "sofa",
#         "14": "escalator",
#         "15": "tv",
#         "16": "painting",
#         "17": "window",
#         "18": "curtain",
#         "19": "fences",
#         "20": "stair",
#         "21": "trunk",
#         "22": "trash",
#         "23": "vase",
#         "24": "babychair"
#     }
color_map_25_dict = {   
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
    25:[255, 255, 255]
}

import numpy as np

def mask_to_points(mask):
    """
    将二进制图像掩码转换为labelme的points格式
    :param mask: 二进制图像掩码
    :return: labelme的points格式
    """
    points = []
    rows, cols = np.where(mask == 1)
    for row, col in zip(rows, cols):
        points.append([col, row])
    return [points]



# {"name": "babychair", "labelIdx": 25, "color": [0, 0, 255], "points":
# "19": {"size": [500, 500], "counts":   序号要+1
# "name": "backgroud", "labelIdx": 1, "color": [220, 20, 60], "points":
# 读取SSA格式的JSON文件
json_path = "/home/data/yang_file/label_utils/1_2.jpg_semantic.json"
with open(json_path, 'r') as j:
    json_content = json.loads(j.read())

# 创建EISEG格式的JSON文件结构
eiseg_data = []
for id_str, mask in json_content['semantic_mask'].items():
    labelidx_int = ade20k_class_remap[int(id_str)]
    if labelidx_int==255:
        labelidx_int = 25
    labelidx_add1_int = labelidx_int+1
    # print(labelidx_add1_int)
    # print(id_str)
    mask_binary = mask_util.decode(mask)
    points = mask_to_points(mask_binary)
    print(points)
  # cv2.findContours会检测出一些多余的点，设定一个阈值
    shape = {
        "name": id2label_25_dict[labelidx_int],
        "labelIdx": labelidx_add1_int,
        "color": color_map_25_dict[labelidx_int],
        "points": points
    }
    eiseg_data.append(shape)

with open('label/1_2.json', 'w') as f:
    json.dump(eiseg_data, f)
  
