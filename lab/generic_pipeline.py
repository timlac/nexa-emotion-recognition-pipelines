import cv2
from pathlib import Path
import mediapipe as mp
from emonet_utils.predict import EmotionPredictor
import csv
import matplotlib.pyplot as plt
import torch


def detect_faces(frame):
    """
    Detects faces in a given frame and returns the bounding boxes.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(frame_rgb)
    bboxes = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            bboxes.append((x, y, w, h))

    return bboxes


def plot_face(face_img, frame_idx, face_id):
    """
    Plots the detected face image.
    """
    plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Frame: {frame_idx}, Face ID: {face_id}')
    plt.axis('off')
    plt.show()


def process_video(video_path: Path, sampling_rate: int, output_csv: Path):
    """
    Processes a video frame-by-frame to detect faces and predict emotions.
    """
    video_capture = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Frame', 'Timestamp', 'Face_ID', 'Predicted_Emotion', 'Valence', 'Arousal'] + list(emotion_predictor.emotion_classes_emonet.values())
        writer.writerow(header)

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_idx % sampling_rate == 0:
                bboxes = detect_faces(frame)
                timestamp = frame_idx / fps

                for face_id, bbox in enumerate(bboxes):
                    x, y, w, h = bbox
                    face_img = frame[y:y + h, x:x + w]

                    # Convert to RGB and resize image to (256, 256)
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    res = emotion_predictor.predict_emotion_from_array(face_img_rgb)

                    print(f"Predicted emotion: {res['predicted_emotion']}, Valence: {res['valence']:.3f}, Arousal: {res['arousal']:.3f}")

                    row = [frame_idx, timestamp, face_id, res['predicted_emotion'], res['valence'], res['arousal']] + list(res['emotion_prob_dict'].values())
                    writer.writerow(row)

                    # plot_face(face_img, frame_idx, face_id)
            frame_idx += 1

    video_capture.release()

# Video path and settings
video_path = Path("../data/videos/A55_gra_v_3_ver1.mp4")
sampling_rate = 1
device = 'cuda:0'
output_csv = Path('../out/emotion_predictions.csv')

mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
emotion_predictor = EmotionPredictor(device)

# Process the video
process_video(video_path, sampling_rate, output_csv)