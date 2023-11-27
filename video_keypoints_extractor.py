import torch
from utils import get_classes_indexes, get_frame_points

import cv2

import os
from tqdm import tqdm

from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

# SELECT CLASSES TO EXTRACT!
classes_to_extract = list({"can", "you", "blow", "my", "whistle", "baby", "whistle", "baby", "let", "me", "know"})
classes_to_extract += ["airplane", "due", "heavy"]

hands_detector = HandDetector()
pose_detector = PoseDetector()

classes_indexes = get_classes_indexes(class_file_name="data/raw/dataset/wlasl_class_list.txt", classes_to_extract=classes_to_extract)

CLASS_TO_IDX = {class_name : idx for (class_name, idx) in  zip(classes_to_extract, classes_indexes)}
IDX_TO_CLASS = {idx : class_name for (class_name, idx) in  zip(classes_to_extract, classes_indexes)}

dataset_path = 'data/internal/preprocessed_videos'
classes_in_dir = [folder_name for folder_name in os.listdir(dataset_path) if folder_name in classes_to_extract]

result_dir = "data/internal/features"
processed_video_names = [os.path.splitext(name)[0] for name in os.listdir(result_dir)]

for class_name in classes_in_dir:
    
    path_to_class = os.path.join(dataset_path, class_name)
    file_names = os.listdir(path_to_class)
    
    bar = tqdm(file_names)
    bar.set_description(class_name)
    
    for file in bar:
        full_path = os.path.join(path_to_class, file)
        video = cv2.VideoCapture(full_path)
        frames_points = []
        frame_cnt = 0
        r_name = os.path.splitext(file)[0]
        
        if r_name in processed_video_names:
            continue

        while video.isOpened():
            ret, frame = video.read()
            frame_cnt += 1
            if ret:
                points = get_frame_points(frame, hands_detector, pose_detector)
                frames_points.append(points)
            elif not ret:
                break

        # Release the video object
        video.release()

        # Convert the list of frames to a PyTorch tensor
        tensor = torch.tensor(frames_points)
        torch.save(tensor, result_dir + "/" + r_name + ".pt")