import cv2
from fer import FER
from deepface import DeepFace


# Initialize the emotion detector
# detector = FER(mtcnn=True)  # Use MTCNN for better face detection

# Initialize OpenCV to capture the pre-recorded video.
video_path = '../data/videos/sentimotion/A65_int_p_3.mov'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)


# # Iterate through video frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Detect emotions on the current frame
#     emotions = detector.detect_emotions(frame)
#
#     # If no emotions are detected, it means no face was found
#     if not emotions:
#         print(f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}, No face detected")
#     else:
#         # Print the detected emotions for each face in the frame
#         for face in emotions:
#             print(f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}, Emotion: {face['emotions']}")
#
# cap.release()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions using DeepFace with a specific model
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        print(result)
        emotion = result[0]['dominant_emotion']
        print(f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}, Dominant Emotion: {emotion}")

    except Exception as e:
        print(f"Error processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}: {str(e)}")

cap.release()