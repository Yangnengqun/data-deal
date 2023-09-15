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
            eiseg_data = []
            for id_str, mask in json_content['semantic_mask'].items():   
                labelidx_int = ade20k_class_remap[int(id_str)+1]
                if labelidx_int==6:
                    labelidx_int = 24
                    labelidx_add1_int = labelidx_int+1
                    # print(labelidx_add1_int)
                    # print(id_str)
                    mask_binary = mask_util.decode(mask)
                    contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获得轮廓

                    # contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for i, contour in enumerate(contours):
                        if hierarchy[0][i][3] == -1:  # 如果没有父轮廓，则保留该轮廓
                        
                            epsilon = 0.01 * cv2.arcLength(contour, True)  # 设置逼近精度
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
           
if __name__ == "__main__":
    # -----------eiseg的json文件保存地址--------------
    # for i in range(1,11):
    #     eiseg_json_root = "/home/data/yang_file/select/baidu/origin/baby_chair_baidu_{}/label".format(i)
    #     if not os.path.exists(eiseg_json_root):
    #         os.makedirs(eiseg_json_root)
    #     # -----------ssa_json_root读取地址----------------
    #     ssa_json_root = "/home/data/yang_file/select/baidu/json_and_mask/baby_chair_baidu_{}".format(i)
    #     print(eiseg_json_root,ssa_json_root)
    #     ssa_to_eiseg(eiseg_json_root, ssa_json_root, min_area = 32)
    
    eiseg_json_root = "/home/data/yang_file/select/baidu/origin/manaul/label"
    if not os.path.exists(eiseg_json_root):
        os.makedirs(eiseg_json_root)
    # -----------ssa_json_root读取地址----------------
    ssa_json_root = "/home/data/yang_file/select/baidu/json_and_mask/manaul"
    print(eiseg_json_root,ssa_json_root)
    ssa_to_eiseg(eiseg_json_root, ssa_json_root, min_area = 32)

