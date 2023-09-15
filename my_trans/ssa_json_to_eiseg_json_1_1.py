import pycocotools.mask as mask_util
import numpy as np
import cv2
import json
import os

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
def ssa_to_eiseg(eiseg_json_root, ssa_json_root, min_area = 16):
    json_names = os.listdir(ssa_json_root)
    for json_name in json_names:
        if ".json" in json_name:
            json_path = os.path.join(ssa_json_root, json_name)
            with open(json_path, 'r') as j:
                json_content = json.loads(j.read())
            # min_area = 16  # 轮廓的最小面积

            # 创建EISEG格式的JSON文件结构
            init_flag = True
            mask_binarys = []
            eiseg_data = []
            for id_str, mask in json_content['semantic_mask'].items():   
                labelidx_int = ade20k_class_remap[int(id_str)]
                if labelidx_int==255:
                    labelidx_int = 25
                if labelidx_int==6:
                    labelidx_int = 24
                labelidx_add1_int = labelidx_int+1
                # print(labelidx_add1_int)
                # print(id_str)                
                mask_binary = mask_util.decode(mask)
                if init_flag:
                    mask_binarys.append()
                    init_flag = False
                # merged_mask[np.where(mask_binary)] *= labelidx_add1_int    
                contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 获得轮廓
                i = 0
                    # contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for i, contour in enumerate(contours): 
                        epsilon = 0.001 * cv2.arcLength(contour, True)  # 设置逼近精度
                        approx = cv2.approxPolyDP(contour, epsilon, True)  # 多边形逼近
                        if len(approx) > 3:  # 过滤掉非三角形的轮廓 
                            area = cv2.contourArea(contour)
                            if area >= min_area:   # cv2.findContours会检测出一些多余的点，设定一个阈值
                                shape = {
                                    "name": id2label_25_dict[labelidx_int],
                                    "labelIdx": labelidx_add1_int,
                                    "color": color_map_25_dict[labelidx_int],
                                    "points": contour.reshape(-1,2).tolist()
                                }
                                eiseg_data.append(shape)
            eiseg_json_path = os.path.join(eiseg_json_root,json_name[:-18]+".json")

            with open(eiseg_json_path, 'w') as f:
                json.dump(eiseg_data, f)
   
# 过滤轮廓形状：使用 cv2.approxPolyDP(contour, epsilon, closed) 函数对轮廓进行多边形逼近，然后根据多边形的边数或角点数来过滤不符合要求的轮廓。
# for contour in contours:
#     epsilon = 0.01 * cv2.arcLength(contour, True)  # 设置逼近精度
#     approx = cv2.approxPolyDP(contour, epsilon, True)  # 多边形逼近
#     if len(approx) > 3:  # 过滤掉非三角形的轮廓
#         filtered_contours.append(contour)   

# 过滤父子轮廓关系：利用轮廓的层级关系 hierarchy 来判断轮廓之间的父子关系，然后根据你的需求筛选出需要的轮廓。
# filtered_contours = []
# for i, contour in enumerate(contours):
#     if hierarchy[0][i][3] == -1:  # 如果没有父轮廓，则保留该轮廓
#         filtered_contours.append(contour)
        
if __name__ == "__main__":
    # -----------eiseg的json文件保存地址--------------
    eiseg_json_root = "/home/data/yang_file/test_y_code/babychair_json_test/baby_chair_baidu_1_json"
    # -----------ssa_json_root读取地址----------------
    ssa_json_root = "/home/data/yang_file/test_y_code/babychair_json_test/baby_chair_baidu_1"
    ssa_to_eiseg(eiseg_json_root, ssa_json_root, min_area = 100)
    
    
    
    
    
# import cv2
# import numpy as np

# # 加载第一个二值图像
# image1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
# ret1, binary1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)

# # 加载第二个二值图像
# image2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)
# ret2, binary2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

# # 进行逻辑与操作
# result = cv2.bitwise_and(binary1, binary2)

# # 判断是否存在前景像素
# overlap = np.any(result)
# if overlap:
#     print("存在重合的前景部分")
# else:
#     print("没有重合的前景部分")


# mask_ = maskUtils.decode(mask)
#                 h, w = mask_.shape
#                 if init_flag:
#                     seg_mask = torch.zeros((1, 1, h, w))
#                     init_flag = False
#                 mask_ = torch.from_numpy(mask_).unsqueeze(0).unsqueeze(0)
#                 seg_mask[mask_] = int(id_str)
#             seg_logit = torch.zeros((1, num_classes, h, w))
#             seg_logit.scatter_(1, seg_mask.long(), 1)
#             seg_logit = seg_logit.float()
#             seg_pred = F.softmax(seg_logit, dim=1).argmax(dim=1).squeeze(0).numpy()
  
