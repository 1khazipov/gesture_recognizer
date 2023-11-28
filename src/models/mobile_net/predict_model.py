import cv2
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import torch
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import optim
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import pickle
import sys

scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))
cap = cv2.VideoCapture(0)
width = 1280
height = 720
width_small = 213
height_small = 120

classes = {
    '0':'fist',
    '1':'ok',
    '2':'rock', 
    '3':'thumb_up'
}

PATH = "models/mobile_net/model20231127_235811_9.pth"
num_classes = 4
threshold = 0.9

cap.set(3, width)
cap.set(4, height)


img_number = 0
offset = 20
img_size = 320
counter = 0

detector = HandDetector(detectionCon=0.8, maxHands=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
#saved_model = pickle.load(open("rf(1).h5","rb"))#
saved_model = models.mobilenet_v3_small()
print(torch.cuda.is_available())
saved_model.classifier[-1] = nn.Linear(1024, num_classes)
saved_model.load_state_dict(torch.load(PATH))
saved_model.to(device)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)


    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
        img_crop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        aspectRatio = h / w
        try:
            if aspectRatio > 1:
                k = img_size / h
                w_calculated = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (w_calculated, img_size))
                w_gap = math.ceil((img_size - w_calculated) / 2)
                img_white[:, w_gap:w_calculated+w_gap] = img_resize
            else:
                k = img_size / w
                h_calculated = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (img_size, h_calculated))
                h_gap = math.ceil((img_size - h_calculated) / 2)
                img_white[h_gap:h_calculated+h_gap, :] = img_resize
                
            cv2.imshow("Crop image", img_white)
            saved_model.eval()
            with torch.no_grad():
                img_white_tensor = torch.Tensor(img_white.transpose(2, 0, 1))
                img_white_tensor /= 255
                img_white_tensor = torch.unsqueeze(img_white_tensor, 0)
                img_white_tensor = img_white_tensor.to(device)
                out = saved_model(img_white_tensor)
                out_class = torch.argmax(out, 1).item()

                if torch.max(out) > threshold:
                    print(classes.get(str(out_class)))
        except Exception:
            print("Out of bound")


    img_small = cv2.resize(img, (width_small, height_small))
    h, w, _ = img.shape
    img[0:height_small, w-width_small:w] = img_small

    
    cv2.imshow("Image_current", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break   