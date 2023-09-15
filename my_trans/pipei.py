# 查看json文件数量是否与mask数量一致，不一致则打印缺少的mask名

import os

# 指定文件所在的目录
directory = "/home/data/yang_file/select/baidu/origin/baby_chair_baidu_8/label"

# 获取目录下的所有文件名
files = os.listdir(directory)

# 将文件名分为PNG文件和JSON文件
png_files = []
json_files = []

for file in files:
    if file.endswith(".png"):
        png_files.append(file[:-4])
    elif file.endswith(".json"):
        json_files.append(file[:-5])

# 将PNG文件名和JSON文件名转换为集合
png_set = set(png_files)
json_set = set(json_files)

# 找出缺少的PNG文件名
missing_png_files = json_set - png_set

# 打印缺少的PNG文件名
for missing_png_file in missing_png_files:
    print("Missing PNG file:", missing_png_file)