import os 
import shutil
# ----------------------------对mask文件重命名，修改后缀为png-------------------------
def rename(png_root):
    if os.path.exists(png_root):
        files = os.listdir(png_root)
        for file in files:   
            if file.endswith(".jpg_semantic.png"):
                newfile = file.replace(".jpg_semantic.png",".png")
                new_file_path = os.path.join(png_root,newfile)
                # new_file_path = os.path.join(png_root,file+"_semantic.png")
                shutil.move(os.path.join(png_root,file),new_file_path)
            elif file.endswith(".png_semantic.png"):
                newfile = file.replace(".png_semantic.png",".png")
                new_file_path = os.path.join(png_root,newfile)
                # new_file_path = os.path.join(png_root,file+"_semantic.png")
                shutil.move(os.path.join(png_root,file),new_file_path)

if __name__ == "__main__":
    png_root = "/home/data/yang_file/data_deal/test_file/test_all/gt_mask"
    rename(png_root)