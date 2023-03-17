import json
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.config import FOCUS_POINTS, mp_drawing, mp_pose, draw_style
from utils.tracker_2d import process_landmarks


MAX_FRAMES = 30
INFO_TEXT = ('Analysing Gesture\n'
             'Window will close automatically\n')

def draw_info(image):
    strokeThikcness = 3
    textThickness = 1
    strokeColor = (0, 0, 0)
    textColor = (255, 255, 255)

    # Split INFO_TEXT on newline characters
    info_text = INFO_TEXT.split('\n')
    for i, line in enumerate(info_text):
        cv2.putText(image, line, (10, 300 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, strokeColor, strokeThikcness,
                    cv2.LINE_AA)
        cv2.putText(image, line, (10, 300 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, textThickness,
                    cv2.LINE_AA)

    return image

def record(gesture_name, file_name):
    # Check if .mov or .mp4 file exists - if it does, choose the right one
    # if not os.path.isfile(path := os.path.join(gesture_name, f"{file_name}.mov")):
    #     path = os.path.join(gesture_name, f"{file_name}.mp4")
    # print("path to video: ", path)

    cap = cv2.VideoCapture(file_name)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diff = num_frames - MAX_FRAMES
    # set the frame position to the middle
    if diff > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, diff // 2)

    history = {num.value: [] for num in FOCUS_POINTS}
    frame = 0

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            frame += 1
            if frame > MAX_FRAMES:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks,  # type: ignore
                                      mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=draw_style)

            if results.pose_world_landmarks:  # type: ignore
                for num in FOCUS_POINTS:
                    landmark = results.pose_world_landmarks.landmark[num.value]  # type: ignore
                    history[num.value].append((landmark.x, landmark.y) if landmark.visibility > 0.7 else (0, 0))

            cv2.imshow('MediaPipe Pose', draw_info(image=cv2.flip(image, 1)))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        time.sleep(3)
        cap.release()
        cv2.destroyAllWindows()

    print('Frames:', len(history[list(history.keys())[0]]))

    return history


name = None


def main(name: str = None, videoFileName: str = None, save_file_name: str = None):
    name = name
    history = record(name, videoFileName)
    processed = process_landmarks(history, plot=False)

    save_json(processed, save_file_name, name)


def save_json(processed, save_file_name: str = None, name = None):
    with open(f'{save_file_name}/{name}.json', 'w') as f:
        json.dump(processed, f, indent=4)


def plot_json():
    with open(f'data/models/gestures/{name}.json', 'r') as f:
        data = json.load(f)

    for landmark_id, points in data.items():
        x, y = zip(*points)

        landmark_id = int(landmark_id)
        color = np.array([landmark_id * 100 % 255, landmark_id * 200 % 255, landmark_id * 300 % 255]) / 255
        plt.plot(x, y, color=color, label=f'Landmark {landmark_id}')

    plt.legend()
    plt.show()


# def compare():
#     with open(f'data/models/gestures/{gesture}.json', 'r') as f:
#         model = json.load(f)
#
#     history = record(gesture, '2')
#     landmark_ids = {int(idx) for idx in model.keys()}
#     processed = process_landmarks(history, include_landmarks=landmark_ids, plot=True)
#
#     distances = []
#     for landmark_id, points in model.items():
#         distance, _ = fastdtw(processed[int(landmark_id)], points, dist=euclidean)
#         # distance = calculate_threshold(points, processed[int(landmark_id)])
#         distances.append(distance)
#
#     print(distances, sum(distances) / len(distances))
