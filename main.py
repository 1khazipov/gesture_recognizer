import time

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

# 21 points on hand
from dataset.asl_dataset import ASLDataset, read_classes, 
read_preprocess_json


def webcum():
    # Play with cv2 and cvzone
    cap = cv2.VideoCapture(0)
    hands_detector = HandDetector()
    pose_detector = PoseDetector()
@@ -24,5 +25,15 @@ def webcum():
        # time.sleep(2)


def load_dataset():
    train, val, test = read_preprocess_json('wlasl_dataset/nslt_100.json', 
'wlasl_dataset/videos')
    classes = read_classes('wlasl_dataset/wlasl_class_list.txt')
    train_dataset = ASLDataset('wlasl_dataset/videos', train, classes)
    return train_dataset


if __name__ == '__main__':
    webcum()
    # webcum()
    dataset = load_dataset()
    print(dataset[0])
