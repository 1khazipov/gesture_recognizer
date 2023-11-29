import json
import os

import sys
scriptpath = "src"
sys.path.append(os.path.abspath(scriptpath))

from utils import get_classes_indexes, extract_frames

# SELECT CLASSES TO EXTRACT!
classes_to_extract = list({"can", "you", "blow", "my", "whistle", "baby", "whistle", "baby", "let", "me", "know"})
classes_to_extract += ["airplane", "due", "heavy"]

classes_data = get_classes_indexes(class_file_name="data/raw/dataset/wlasl_class_list.txt", classes_to_extract=classes_to_extract)

CLASS_TO_IDX = {class_name : idx for (idx, class_name) in  classes_data}
IDX_TO_CLASS = {idx : class_name for (idx, class_name) in  classes_data}

videos = json.load(open("data/raw/dataset/nslt_2000.json"))

source_video_folder = "data/raw/dataset/videos"
destination_folder_path = 'data/internal/preprocessed_videos'

if not os.path.exists(destination_folder_path):
    os.makedirs(destination_folder_path)
for class_name in classes_to_extract:
    dir_name = os.path.join(destination_folder_path, class_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

new_data_file = {}

for key in videos:
    class_idx = int(videos[key]["action"][0])
    if class_idx in IDX_TO_CLASS.keys():
        file_name = key + '.mp4'
        video_path = os.path.join(source_video_folder, key+'.mp4')       
        
        if not os.path.exists(video_path) or not os.path.isfile(video_path):
            continue
        
        start_frame, end_frame = videos[key]["action"][1], videos[key]["action"][2]
        
        destination_class_path = destination_folder_path + '/' + IDX_TO_CLASS[class_idx]
        output_path = os.path.join(destination_class_path, key+'.mp4')
        
        extract_frames(video_path, start_frame, end_frame, output_path)