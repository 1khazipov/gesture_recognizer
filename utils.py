from typing import List, Tuple
import cv2

def get_classes_indexes(class_file_name: str, classes_to_extract: List[str]) -> List[Tuple[int, str]]:
    class_file = open(class_file_name, "r")
    class_data = class_file.read().split("\n")
    class_file.close()

    classes_data = []

    for i in range(0, len(class_data)-1):
        line = class_data[i]
        pair = line.strip().split('\t')
        index = int(pair[0])
        value = pair[1]
        if value in classes_to_extract:
            classes_data.append((index, value))

    return classes_data


def get_frame_points(frame, hands_detector, pose_detector):
    # Collect all points. 21 points for each hand, 33 points on pose
    points = [0] * (21 * 3 * 2 + 33 * 3)
    # Recognize hands and collect them into list of all points
    hands, img1 = hands_detector.findHands(frame)
    for i in range(len(hands)):
        ind_shift = 0
        if hands[i].get('type') == 'Left':
            ind_shift = 21 * 3
        hand_points = hands[i].get('lmList')
        for j in range(len(hand_points)):
            for k in range(3):
                points[ind_shift + j * 3 + k] = hand_points[j][k]

    # Recognize the pose and collect points
    img2 = pose_detector.findPose(frame)
    lmList, bboxInfo = pose_detector.findPosition(frame, bboxWithHands=False)
    for i in range(len(lmList)):
        for j in range(3):
            points[21 * 3 * 2 + i * 3 + j - 1] = lmList[i][j]

    return points

def extract_frames(input_video_path : str, start_frame : int, end_frame : int, output_path : str) -> None:
        # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)

    # Get frames per second (fps) and total number of frames in the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure start_frame and end_frame are within the valid range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))

    # Set the video capture to the start_frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (int(video_capture.get(3)), int(video_capture.get(4))))

    # Loop through frames and write to the output video
    frame_number = start_frame
    while frame_number <= end_frame:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Write the frame to the output video
        output_video.write(frame)

        frame_number += 1

    # Release the video capture and writer objects
    video_capture.release()
    output_video.release()