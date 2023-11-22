import os
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def make_grid_image(image_list,col_num):
    img_h,img_w = image_list[0].shape[:2]  # 图片尺寸（假设所有图片尺寸相同）

    # 计算输出图像的宽度和高度
    output_width = img_w * col_num
    output_height = ((len(image_list) - 1) // col_num + 1) * img_h

    # 创建一个空白的输出图像
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # 循环遍历图片列表
    for i, image in enumerate(image_list):
        # 计算当前图片的行索引和列索引
        row = i // col_num
        col = i % col_num
        
        # 将当前图片复制到输出图像的对应位置
        output_image[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w, :] = image

    return output_image


img_form = "*.png"
        
def make_viz_video(frame_path, vid_path):
    frame_path = os.path.join(frame_path,img_form)
    CMD_STR = 'ffmpeg -framerate 10 -pattern_type glob -i "{}" "{}"  -nostats -loglevel 0 -y'.format(frame_path, vid_path)
    print(CMD_STR)
    os.system(CMD_STR)
    time.sleep(1) # wait 10 seconds

def add_superscript(superscript_text,image):
    changed_image = image.copy()
    img = Image.fromarray(changed_image)
    # 设置上标文本和字体
    font_size = 30
    # font = ImageFont.truetype("times.ttf", font_size)
    font = ImageFont.load_default()

    # 计算上标文本的宽度和高度
    text_width, text_height = font.getsize(superscript_text)

    # 在图像上创建一个新的绘图对象
    draw = ImageDraw.Draw(img)

    # 设置上标文本的位置（假设在左上角偏移10个像素）
    x = 10
    y = 10

    # 绘制一个矩形作为上标的背景
    rectangle_width = text_width + 10
    rectangle_height = text_height + 10
    rectangle_coords = [(x, y), (x + rectangle_width, y + rectangle_height)]
    draw.rectangle(rectangle_coords, fill=(255, 255, 255))

    # 在矩形内绘制上标文本
    draw.text((x + 5, y + 5), superscript_text, fill=(0, 0, 0), font=font)
    # 通道已经被Image变了
    # image_hwc = np.transpose(changed_image, (1, 0, 2))
    # breakpoint()
    return changed_image


def transfer_frames_in_subdirectories(root_folder, object_dir):
    if not os.path.exists(object_dir):
        os.makedirs(object_dir)
    # 遍历根文件夹
    for subfolder in os.listdir(root_folder):
        # 遍历子目录
        subfolder_path = os.path.join(root_folder, subfolder)
        # 遍历目录中的文件
        objfolder_path = os.path.join(object_dir, subfolder+".mp4")

        make_viz_video(subfolder_path, objfolder_path)

def video_image(folder_dict:dict,output_dir,vid_dirname,col_num=2):
    first_key = next(iter(folder_dict))
    first_folder = folder_dict[first_key]

    for image_name in sorted(os.listdir(first_folder)):
        image_list = []
        if not image_name.endswith(".png") :
            continue
        for key,folder in folder_dict.items():
            image_path = os.path.join(folder, image_name)
            _f = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(_f.shape)==2:
                _f = cv2.cvtColor(_f, cv2.COLOR_GRAY2BGR)
            sc_f = add_superscript(key,_f)
            # breakpoint()
            image_list.append(sc_f)
        grid_image = make_grid_image(image_list,col_num)
        output_path = os.path.join(output_dir,image_name)
        cv2.imwrite(output_path,grid_image)
    make_viz_video(output_dir,vid_dirname)

if __name__ == "__main__":
    # 把图片转为视频
    # root_folder = "/home/liurui/DATA/result/cutie/Annotations"
    # object_dir = "/home/liurui/DATA/result/viz/cutie"
    # transfer_frames_in_subdirectories(root_folder,object_dir)
    # 打成网格转为视频
    folder_dict={
        "masks":"/home/liurui/DATA/realhuman_reformat/production ID_5197613/masks",
        "frames":"/home/liurui/DATA/realhuman_reformat/production ID_5197613/frames",
        "cutie":"/home/liurui/DATA/realhuman_reformat/production ID_5197613/cutie-result"
    }
    out_dir = "/home/liurui/DATA/tmp/composite"
    video_dirname = "/home/liurui/DATA/tmp/composite.mp4"
    video_image(folder_dict,out_dir,video_dirname)