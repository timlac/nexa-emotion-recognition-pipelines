import cv2
import mediapipe as mp

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
            confidence_score = detection.score[0]  # Extract the confidence score

            # Ensure bounding box is within frame boundaries
            if x < 0 or y < 0 or x + w > iw or y + h > ih:
                print(f"Bounding box {(x, y, w, h, confidence_score)} is out of frame boundaries. Skipping.")
                continue

            bboxes.append((x, y, w, h, confidence_score))

    return bboxes