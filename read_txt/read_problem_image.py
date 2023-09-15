# file_path = "/home/data/yang_file/data_deal/txt/problem_image.txt"
# lis_all = []
# with open(file_path, "r") as file:
#     lines = file.readlines()
#     for line in lines:
#         lis_all.append(line.strip())  # 使用 strip() 方法去除行尾的换行符
# print(len(lis_all))
 

# -------------------------------读取lst文件，并将lst的图片保存在指定位置---------------------
import shutil
import os
def copy_problem_image(txt_imagename_path,image_dir,out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(txt_imagename_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            file_path = os.path.join(image_dir,line.strip())
            if os.path.exists(out_dir):
                shutil.copy(file_path, out_dir)

if __name__=="__main__":
    txt_imagename_path = "/home/data/yang_file/data_deal/txt/problem_image.txt"
    image_dir = "/home/data/datasets/Map_V10_DB10_auto_label_datasets_deal/ADE20K/images"
    out_dir = "/home/data/yang_file/data_deal/problem_image/ADE20K/images"
    copy_problem_image(txt_imagename_path,image_dir,out_dir)
    
