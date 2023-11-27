import shutil
import os

from utils import get_classes_indexes

# SELECT CLASSES TO EXTRACT!
classes_to_extract = list({"can", "you", "blow", "my", "whistle", "baby", "whistle", "baby", "let", "me", "know"})
classes_to_extract += ["airplane", "due", "heavy"]

classes_data = get_classes_indexes(class_file_name="data/raw/dataset/wlasl_class_list.txt", classes_to_extract=classes_to_extract)

CLASS_TO_IDX = {class_name : idx for (idx, class_name) in  classes_data}
IDX_TO_CLASS = {idx : class_name for (idx, class_name) in  classes_data}

source_folder_path = "data/raw/custom_video_dataset"
destination_folder_path = 'data/internal/preprocessed_videos'

for class_name in classes_to_extract:
    dir_name = os.path.join(destination_folder_path, class_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

classes_to_extract = list(set.union(set(os.listdir(source_folder_path)), set(classes_to_extract)))

for class_name in classes_to_extract:
    source_class_path = source_folder_path + '/' + class_name
    destination_class_path = destination_folder_path + '/' + class_name
    
    if not os.path.exists(source_class_path):
        os.makedirs(source_class_path)
        
    for file in os.listdir(source_class_path):
        file_path = os.path.join(source_class_path, file)
        shutil.copy(file_path, destination_class_path)