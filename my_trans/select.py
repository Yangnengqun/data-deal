# 删除了一些图片，需要将json文件、灰度图对应,要删除一些多余的json文件、灰度图
import os


# for i in range(1,11):
#     # 指定图片和json文件所在的目录
#     image_dir = "/home/data/yang_file/select/baidu/origin/baby_chair_baidu_{}".format(i)
#     json_and_mask_dir = "/home/data/yang_file/select/baidu/json_and_mask/baby_chair_baidu_{}".format(i)

#     # 获取所有图片文件名
#     json_and_mask_files = os.listdir(json_and_mask_dir)

#     # 遍历图片文件
#     for json_and_mask_file in json_and_mask_files:
#         # 获取图片文件名（不包含扩展名）
#         json_and_mask_name, extension = os.path.splitext(json_and_mask_file)      
#         # 构建对应的image文件路径
#         if extension ==".json":
#             image_file = os.path.join(image_dir, json_and_mask_name[:-13]+'.jpg')
#             print(image_file)
#             if not os.path.exists(image_file):
#                 json_file = os.path.join(json_and_mask_dir, json_and_mask_file)
#                 if os.path.exists(json_file):
#                     os.remove(json_file)
#         elif extension ==".png":
#             image_file = os.path.join(image_dir, json_and_mask_name[:-13]+'.jpg')
#             print(image_file)
#             if not os.path.exists(image_file):
#                 mask_file = os.path.join(json_and_mask_dir, json_and_mask_file)
#                 if os.path.exists(mask_file):
#                     os.remove(mask_file)
    
    
    
# 指定图片和json文件所在的目录
i = 8
image_dir = "/home/data/yang_file/select/baidu/origin/baby_chair_baidu_{}".format(i)
json_and_mask_dir = "/home/data/yang_file/select/baidu/origin/baby_chair_baidu_{}/label".format(i)


# 获取所有图片文件名
json_and_mask_files = os.listdir(json_and_mask_dir)

# 遍历图片文件
for json_and_mask_file in json_and_mask_files:
    # 获取图片文件名（不包含扩展名）
    json_and_mask_name, extension = os.path.splitext(json_and_mask_file)      
    # 构建对应的image文件路径
    if extension ==".json":
        image_file = os.path.join(image_dir, json_and_mask_name+'.jpg')
        if not os.path.exists(image_file):
            print(json_and_mask_name)
            json_file = os.path.join(json_and_mask_dir, json_and_mask_file)
            # if os.path.exists(json_file):
            #     os.remove(json_file)
    elif extension ==".png":
        image_file = os.path.join(image_dir, json_and_mask_name+'.jpg')
        if not os.path.exists(image_file):
            print(json_and_mask_name)
            mask_file = os.path.join(json_and_mask_dir, json_and_mask_file)
            # if os.path.exists(mask_file):
            #     os.remove(mask_file)