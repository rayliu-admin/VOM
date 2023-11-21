import os
import shutil

def copy_and_rename_folders(source_folder, destination_folder,new_folder_name):
    # 获取源文件夹中的所有文件夹
    folders = next(os.walk(source_folder))[1]
    
    # 遍历每个文件夹
    for folder_name in folders:
        # 构建源文件夹的完整路径
        source_subfolder = os.path.join(source_folder, folder_name)
        
        # 如果当前文件夹存在
        if os.path.isdir(source_subfolder):
            # 构建目标文件夹的新名称和完整路径
            destination_subfolder = os.path.join(destination_folder, folder_name,new_folder_name)
            print(destination_subfolder)
            if os.path.exists(destination_subfolder):
                # 删除文件夹及其内容
                shutil.rmtree(destination_subfolder)
            # 复制文件夹并更名
            shutil.copytree(source_subfolder, destination_subfolder)

new_folder_name = "cutie-result"         
# 源文件夹的路径
source_folder = "/home/liurui/DATA/result/cutie/Annotations"
# 目标文件夹的路径
destination_folder = "/home/liurui/DATA/realhuman_reformat"

# 调用函数进行文件夹复制和重命名
copy_and_rename_folders(source_folder, destination_folder,new_folder_name)