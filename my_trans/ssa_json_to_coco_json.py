import json
import os
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
# json_path = "/home/data/yang_file/label_utils/1_2.jpg_semantic.json"
# # 读取JSON文件
# with open('json_path', 'r') as f:
#     data = json.load(f)

# # 创建COCO格式的JSON文件结构
# coco_data = {
#     "categories": [],
#     "images": [],
#     "annotations": [],
#     "info": {},
#     "licenses": []
# }

# # 添加图像信息到COCO JSON文件
# for i, image in enumerate(data['images']):
#     img = Image.open(os.path.join(os.path.dirname('input.json'), image['file_name']))
#     width, height = img.size

#     coco_image = {
#         "id": i,
#         "width": width,
#         "height": height,
#         "file_name": image['file_name']
#     }

#     coco_data['images'].append(coco_image)

# # 添加类别信息到COCO JSON文件
# for category in data['categories']:
#     coco_category = {
#         "id": category['id'],
#         "name": category['name']
#     }

#     coco_data['categories'].append(coco_category)

# # 添加注释信息到COCO JSON文件
# annotation_id = 0
# for annotation in data['annotations']:
#     # 获取物体类别和掩码信息
#     category_id = annotation['category_id']
#     rle = annotation['segmentation']

#     # 将RLE掩码转换为二进制掩码
#     mask = mask_util.decode(rle)

#     # 添加注释信息到COCO JSON文件
#     coco_annotation = {
#         "id": annotation_id,
#         "image_id": annotation['image_id'],
#         "category_id": category_id,
#         "iscrowd": 0,
#         "area": int(mask_util.area(rle)),
#         "bbox": mask_util.toBbox(rle).tolist(),
#         "segmentation": mask_util.encode(np.asfortranarray(mask)).tolist()
#     }

#     coco_data['annotations'].append(coco_annotation)
#     annotation_id += 1

# # 将COCO JSON文件保存到磁盘
# with open('output.json', 'w') as f:
#     json.dump(coco_data, f)


json_path = "/home/data/yang_file/label_utils/1_2.jpg_semantic.json"



# {"name": "babychair", "labelIdx": 25, "color": [0, 0, 255], "points":
# "19": {"size": [500, 500], "counts":   序号要+1
# "name": "backgroud", "labelIdx": 1, "color": [220, 20, 60], "points":
# 读取SSA格式的JSON文件
json_path = "/home/data/yang_file/label_utils/1_2.jpg_semantic.json"
with open(json_path, 'r') as j:
    json_content = json.loads(j.read())

import json
import numpy as np
import pycocotools.mask as mask_util

# 定义RLE掩码
rle = {'counts': 'eNqNjV2Pk0AURff8Rb3fJWZzQ6aGQNUgZtL3lq5y7dG4U9UxUJm7n3P9z3D3fV2vHmCjMw5ZjGwzZ9+7s6X7fDv+T7+7+8fHv7+3+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+7+a5fcg==', 'size': [480, 640]}

# 将RLE掩码解码为二进制掩码
mask = mask_util.decode(rle)

# 构造JSON格式的注释信息
annotation = {
    "id": 1,
    "image_id": 1,
    "category_id": 2,
    "iscrowd": 0,
    "area": int(mask_util.area(rle)),
    "bbox": mask_util.toBbox(rle).tolist(),
    "segmentation": mask_util.encode(np.asfortranarray(mask)).tolist()
}

# 打印JSON格式的注释信息
print(json.dumps(annotation))
