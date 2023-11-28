import torch

import sys
import os
import cv2
import argparse

from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))
from src.utils import get_frame_points, get_classes_indexes
from src.models.utils import get_device, get_dataset, set_seed
from src.models.complex_lstm.train_model import ComplexLSTM

def main(args):
    full_path = args.file_path
    
    set_seed()
    _, _, idx_to_class = get_dataset()
    
    source_folder = os.path.join('data', 'internal', 'preprocessed_videos')
    classes_to_extract = os.listdir(source_folder)
    classes_data = get_classes_indexes(class_file_name="data/raw/dataset/wlasl_class_list.txt", classes_to_extract=classes_to_extract)
    CLASS_IDX_TO_CLASS_NAME = {idx : class_name for (idx, class_name) in  classes_data}
    
    hands_detector = HandDetector()
    pose_detector = PoseDetector()
    
    video = cv2.VideoCapture(full_path)
    frames_points = []
    frame_cnt = 0
    
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
    frames_tensor = torch.tensor([frames_points], dtype=torch.float)
    
    model = ComplexLSTM(input_size=225, output_size=len(idx_to_class))
    
    checkpoint = torch.load('models/complex_lstm/best.pt')
    
    model.load_state_dict(checkpoint['model_state_dict'])

    outputs = model(frames_tensor)

    # Get predicted classes and true classes from data            
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    prediction = torch.argmax(probabilities, dim=0).item()
    prediction_idx = idx_to_class[prediction]
    prediction_label = CLASS_IDX_TO_CLASS_NAME[prediction_idx]
    
    print('Prediction:', prediction_label)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help='Path to a video file to predict gesture')

    args = parser.parse_args()

    main(args)