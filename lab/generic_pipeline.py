import cv2
from pathlib import Path
import mediapipe as mp
from emonet_utils.predict import EmotionPredictor
import csv
import matplotlib.pyplot as plt
import torch
import os


# TODO: Add options for processing videos with multiple faces,
# In the KOSMOS study we are only interested in the leftmost face detected in each frame.
# TODO: Implement a function to investigate how many faces are detected in each frame. Alert for anomalies.
# TODO: Add HSemotion option to the pipeline.
# TODO: Possibly add even more models for emotion recognition



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
            confidence_score = detection.score[0]  # Extract the confidence score
            bboxes.append((x, y, w, h, confidence_score))

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
        header = ['Frame', 'Timestamp', 'Face_ID', 'Confidence_Score', 'Predicted_Emotion', 'Valence', 'Arousal'] + list(emotion_predictor.emotion_classes_emonet.values())
        writer.writerow(header)

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_idx % sampling_rate == 0:
                bboxes = detect_faces(frame)
                timestamp = frame_idx / fps
                # print("processing frame", frame_idx)

                print(f"Detected {len(bboxes)} faces in frame {frame_idx}")
                # if len(bboxes) != 2:
                #     print(f"Weird number of faces detected: {len(bboxes)} for frame {frame_idx}... skipping")
                #     continue

                # Sort bounding boxes by x-coordinate (left to right)
                bboxes.sort(key=lambda item: item[0])

                for face_id, bbox in enumerate(bboxes):

                    # if face_id > 0:
                    #     continue

                    x, y, w, h, confidence_score  = bbox
                    face_img = frame[y:y + h, x:x + w]

                    # plot_face(face_img, frame_idx, face_id)

                    # Convert to RGB and resize image to (256, 256)
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    res = emotion_predictor.predict_emotion_from_array(face_img_rgb)

                    # print(f"Predicted emotion: {res['predicted_emotion']}, Valence: {res['valence']:.3f}, Arousal: {res['arousal']:.3f}")

                    row = [frame_idx, timestamp, face_id, confidence_score, res['predicted_emotion'], res['valence'], res['arousal']] + list(res['emotion_prob_dict'].values())
                    writer.writerow(row)

                    # plot_face(face_img, frame_idx, face_id)
            frame_idx += 1

    video_capture.release()


# video_path = Path('/home/tim/.sensitive_data/kosmos/split/KOSMOS077_IS_LSI_LEFT.mp4')
# video_path = Path('../data/videos/sentimotion/A55_hap_p_3.mp4')
video_dir = Path('/home/tim/.sensitive_data/kosmos/split')

# Iterate over all video files in the directory
for video_file in video_dir.glob('*.mp4'):
    filename = video_file.stem
    sampling_rate = 10000
    device = 'cuda:0'
    output_csv = Path(f'../out/emotion_predictions_{filename}.csv')

    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    emotion_predictor = EmotionPredictor(device)

    # Process the video
    process_video(video_file, sampling_rate, output_csv)