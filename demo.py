import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

import torch.nn as nn
import torch

from src.utils import get_frame_points
from src.models.simple_lstm.train_model import SimpleLSTM
from src.models.complex_lstm.train_model import ComplexLSTM

import argparse
    
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


def webcam(model, device, IDX_TO_CLASS, CLASS_IDX_TO_CLASS_NAME, threshold):
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
            if (probabilities[prediction] >= threshold):
                prediction_idx = IDX_TO_CLASS[prediction]
                prediction_label = CLASS_IDX_TO_CLASS_NAME[prediction_idx]
                cv2.putText(img_flipped, prediction_label, (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("CAM", img_flipped)
            
        key = cv2.waitKey(1)
        if key==ord('q'):
            break

def main(args):
    source_folder = os.path.join('data', 'internal', 'preprocessed_videos')

    classes_to_extract = os.listdir(source_folder)
    classes_data = get_classes_indexes(class_file_name="data/raw/dataset/wlasl_class_list.txt", classes_to_extract=classes_to_extract)
    dataset = create_dataframe(source_folder, classes_data)
    CLASS_IDX_TO_CLASS_NAME = {idx : class_name for (idx, class_name) in  classes_data}

    IDX_TO_CLASS = [ c_id for i, c_id in enumerate(set(dataset.class_id))]
    CLASS_TO_IDX = { c_id: i for i, c_id in enumerate(IDX_TO_CLASS)}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if args.model_name == 'simple_lstm':
        model = SimpleLSTM(input_size=225, output_size=len(IDX_TO_CLASS))
    if args.model_name == 'complex_lstm':
        model = ComplexLSTM(input_size=225, output_size=len(IDX_TO_CLASS))
    
    load_ckpt_path = f'models/{args.model_name}'
    model_ckpt_path = load_ckpt_path + '/' + args.checkpoint_name

    checkpoint = torch.load(model_ckpt_path) if torch.cuda.is_available() else torch.load(model_ckpt_path, map_location=lambda storage, loc: storage) 

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
            
    webcam(model, device, IDX_TO_CLASS, CLASS_IDX_TO_CLASS_NAME, threshold)

if __name__ == '__main__':
    checkpoint_name = 'best.pt'
    model_name = 'simple_lstm'
    threshold = 0.9

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", nargs='?', const=checkpoint_name, type=str, default=checkpoint_name, help='Name of the checkpoint to get results from')
    parser.add_argument('--model_name', nargs='?', const=model_name, type=str, default=model_name, help='Name of the model to use')
    parser.add_argument('--threshold', nargs='?', const=threshold, type=float, default=threshold, help='Inference threshold')

    args = parser.parse_args()

    main(args)