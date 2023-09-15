import os
import shutil

# 源文件夹路径
source_folder = '/home/data/yang_file/Map_V10_DB10/bbchair/offical_chair_image/self-chair'

# 目标文件夹路径
target_folder = '/home/data/yang_file/Map_V10_DB10/bbchair/offical_chair_image/self-chair-merge'
for i,(root, dirs, files) in enumerate(os.walk(source_folder)):
    print(root, dirs, files)
    for index, file_name in enumerate(files):
        # 仅处理图片文件
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # 构建新文件名
            new_file_name = f'self-chair-{i}_{index+1}'
            
            # 源文件路径
            source_file_path = os.path.join(root, file_name)
            
            # 目标文件路径
            target_file_path = os.path.join(target_folder, f'{new_file_name}.jpg')
            
            # 复制并重命名文件
            shutil.copyfile(source_file_path, target_file_path)
            
            # 打印文件复制信息
            # print(f'Copied: {source_file_path} -> {target_file_path}')