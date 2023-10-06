import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

def webcam():
    # Play with cv2 and cvzone
    cap = cv2.VideoCapture(0)
    hands_detector = HandDetector()
    pose_detector = PoseDetector()
    while True:
        success, img = cap.read()
        hands, img = hands_detector.findHands(img)
        img = pose_detector.findPose(img)
        lmList, bboxInfo = pose_detector.findPosition(img, bboxWithHands=False)
        cv2.imshow("CAM", img)
        cv2.waitKey(1)

        # print(lmList)
        # print(bboxInfo)
        # print()
        # time.sleep(2)


if __name__ == '__main__':
    webcam()