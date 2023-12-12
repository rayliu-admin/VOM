import os
import shutil

def copy_and_rename_folders(input_folder_path, output_folder_path):
    # 获取输入文件夹中的图片文件名列表
    input_image_names = os.listdir(input_folder_path)
    input_image_names = sorted(input_image_names)

    # 获取输出文件夹中的图片文件名列表
    output_image_names = os.listdir(output_folder_path)
    output_image_names = sorted(output_image_names)

    # 确保两个文件夹中的图片数量相同
    if len(input_image_names) == len(output_image_names):
        # 遍历图片文件名列表，依次修改输出文件夹中的图片名字
        for i in range(len(input_image_names)):
            input_image_name = input_image_names[i]
            output_image_name = output_image_names[i]

            # 构建完整的文件路径
            input_image_path = os.path.join(input_folder_path, input_image_name)
            output_image_path = os.path.join(output_folder_path, output_image_name)

            # 修改输出文件夹中的图片名字
            os.rename(output_image_path, os.path.join(output_folder_path, input_image_name))
            print(f"已将 {output_image_name} 修改为 {input_image_name}")
    else:
        print("错误：两个文件夹中的图片数量不相同。")

data_dir = "/home/liurui/DATA/realhuman_reformat"
for dirnames in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir,dirnames)):
        continue
    rvm_result_dir = os.path.join(data_dir,dirnames, 'rvm-result')
    alphas_result_dir = os.path.join(data_dir,dirnames, 'alphas')
    copy_and_rename_folders(alphas_result_dir,rvm_result_dir)