import os
import time

img_form = "*.png"
        
def make_viz_video(frame_path, vid_path):
    frame_path = os.path.join(frame_path,img_form)
    CMD_STR = 'ffmpeg -framerate 10 -pattern_type glob -i "{}" "{}"  -nostats -loglevel 0 -y'.format(frame_path, vid_path)
    print(CMD_STR)
    os.system(CMD_STR)
    time.sleep(1) # wait 10 seconds


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

if __name__ == "__main__":
    root_folder = "/home/liurui/DATA/result/cutie/Annotations"
    object_dir = "/home/liurui/DATA/result/viz/cutie"
    transfer_frames_in_subdirectories(root_folder,object_dir)
