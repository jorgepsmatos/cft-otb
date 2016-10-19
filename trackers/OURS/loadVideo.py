import numpy as np
from os import listdir
from os.path import isfile, join


# LOADS VIDEO INFO FROM FOLDER AND TXT FILES
def load_video_info(video_path):
    txt_file = 'groundtruth_rect.txt'
    groundtruth_file = video_path + '/' + txt_file
    groundtruth = np.loadtxt(groundtruth_file, delimiter=",")
    init_pos = groundtruth[0, :-2]
    init_pos = init_pos[::-1]
    target_sz = groundtruth[0, 2:]
    target_sz = target_sz[::-1]
    video_path = video_path + '/' + 'imgs/'
    img_files = [f for f in listdir(video_path) if isfile(join(video_path, f))]
    img_files.sort()
    return img_files, init_pos, target_sz, groundtruth, video_path
