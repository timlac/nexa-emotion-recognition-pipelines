import cv2
import mediapipe as mp
import csv
from hsemotion.facial_emotions import HSEmotionRecognizer
from tqdm import tqdm  # Progress bar
import time
import torch.nn.functional as F
import torch


# Initialize MediaPipe for fast face detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize HSEmotionRecognizer (Use 'cuda' for GPU or 'cpu' for CPU)
model_name = 'enet_b0_8_va_mtl'  # Model supporting valence-arousal prediction
emotion_recognizer = HSEmotionRecognizer(model_name=model_name, device='cuda')

# Video capture (you can change '0' to a file path for a video file)
video_path = "../data/videos/998f4f69-889b-4b20-b6a8-5a3e2b05a565.mkv"
cap = cv2.VideoCapture(video_path)

# Frame sampling settings
sampling_rate = 20  # Process every 10th frame to reduce computation for long videos
frame_count = 0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Store predictions in a list
predictions = []

# Define emotion classes based on HSEmotion
emotion_classes = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise', 'Valence',
                   'Arousal']

# CSV file to save results
output_csv = '../out/emotion_predictions_with_probabilities.csv'
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header: Frame, Face_ID, then all emotions, valence, and arousal
    header = ['Frame', 'Face_ID'] + emotion_classes
    writer.writerow(header)

    # Set up progress bar
    with tqdm(total=total_frames // sampling_rate, desc="Processing Video Frames") as pbar:
        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            frame_count += 1

            # Process every 'sampling_rate' frame
            if frame_count % sampling_rate == 0:
                # Convert the frame to RGB, as required by MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Face detection using MediaPipe
                results = mp_face_detection.process(frame_rgb)

                if results.detections:
                    for face_id, detection in enumerate(results.detections):
                        # Get the bounding box around the face
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(
                            bboxC.height * ih)

                        # Crop the face from the frame
                        face_img = frame[y:y + h, x:x + w]

                        # Emotion recognition on the cropped face
                        results = emotion_recognizer.predict_emotions(face_img, logits=True)

                        logits = torch.tensor(results[1])

                        # Apply softmax to the logits for emotions (first 8 values)
                        emotion_logits = logits[:8]
                        emotion_probabilities = F.softmax(emotion_logits, dim=0)

                        # Valence and Arousal (last 2 values, no softmax)
                        valence = logits[8].item()
                        arousal = logits[9].item()

                        # Collect the probabilities for each emotion and log them
                        row = [frame_count, face_id] + [emotion_probabilities[i].item() for i in range(8)] + [valence, arousal]

                        # Save prediction to CSV
                        writer.writerow(row)

                # Update progress bar
                pbar.update(1)

# Clean up
cap.release()

print(f"Emotion predictions saved to {output_csv}")
