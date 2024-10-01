import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from pathlib import Path

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.1)

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

def plot_frame_with_bboxes(frame, bboxes):
    """
    Plots the frame with bounding boxes for detected faces.
    """
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def process_video(video_path: Path):
    """
    Processes a video frame-by-frame to detect faces and plot the frame with bounding boxes.
    """
    video_capture = cv2.VideoCapture(str(video_path))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        bboxes = detect_faces(frame)
        plot_frame_with_bboxes(frame, bboxes)

    video_capture.release()

# Path to the video file
video_path = Path('/home/tim/.sensitive_data/kosmos/split/KOSMOS020_RMW_BAS_LEFT.mp4')

# Process the video
process_video(video_path)