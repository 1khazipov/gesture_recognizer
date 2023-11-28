import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

import torch.nn as nn
import torch

from src.utils import get_frame_points

THRESHOLD = 0.95

class SequenceModel(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.lstm = nn.Sequential(
            nn.LSTM(input_size, 64, 1, batch_first=True, bidirectional=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 2, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])

        return x
    
import os
from src.utils import get_classes_indexes
from pathlib import Path
import pandas as pd
def create_dataframe(videos_root, classes_data):
    data = []

    CLASS_TO_IDX = {class_name : idx for (idx, class_name) in classes_data}
    
    for class_name in os.listdir(videos_root):
        class_path = os.path.join(videos_root, class_name)
        for name in os.listdir(class_path):
            name = Path(name).stem
            data.append([name, CLASS_TO_IDX[class_name], class_name])
    return pd.DataFrame(data, columns=['video_name', 'class_id', 'class_name'])


def webcam(model, device, IDX_TO_CLASS, CLASS_IDX_TO_CLASS_NAME):
    # Play with cv2 and cvzone
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    hands_detector = HandDetector()
    pose_detector = PoseDetector()
    
    frames_points = []
    
    while True:
        ret, img = cap.read()
        img_flipped = cv2.flip(img, 1)
        # hands, img = hands_detector.findHands(img_filpped)
        # img = pose_detector.findPose(img)

        if ret:
            points = get_frame_points(img_flipped, hands_detector, pose_detector)
            frames_points = [*frames_points[:119], points]
            frames_tensor = torch.tensor([frames_points]).to(device, dtype=torch.float)
            
            outputs = model(frames_tensor)

            # Get predicted classes and true classes from data            
            probabilities = nn.functional.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(probabilities, dim=0).item()
            if (probabilities[prediction] >= THRESHOLD):
                prediction_idx = IDX_TO_CLASS[prediction]
                prediction_label = CLASS_IDX_TO_CLASS_NAME[prediction_idx]
                cv2.putText(img_flipped, prediction_label, (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("CAM", img_flipped)
            
        key = cv2.waitKey(1)
        if key==ord('q'):
            break

if __name__ == '__main__':
    source_folder = os.path.join('data', 'internal', 'preprocessed_videos')

    classes_to_extract = os.listdir(source_folder)
    classes_data = get_classes_indexes(class_file_name="data/raw/dataset/wlasl_class_list.txt", classes_to_extract=classes_to_extract)
    dataset = create_dataframe(source_folder, classes_data)
    CLASS_IDX_TO_CLASS_NAME = {idx : class_name for (idx, class_name) in  classes_data}

    IDX_TO_CLASS = [ c_id for i, c_id in enumerate(set(dataset.class_id))]
    CLASS_TO_IDX = { c_id: i for i, c_id in enumerate(IDX_TO_CLASS)}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
            
    model = SequenceModel(input_size=225, output_size=len(IDX_TO_CLASS))
    model.load_state_dict(torch.load('models/simple_lstm/best.pt'))
    model.eval()
    model.to(device)
            
    webcam(model, device, IDX_TO_CLASS, CLASS_IDX_TO_CLASS_NAME)