import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize OpenCV to capture the pre-recorded video.
video_path = '../data/A65_ang_p_2.mov'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Open a CSV file to write the landmarks data
with open('landmarks_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header row
    writer.writerow(['frame', 'landmark_index', 'x', 'y', 'z'])

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image color to RGB as Mediapipe expects RGB input.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        # Extract facial landmarks and write them to the CSV file.
        if result.multi_face_landmarks:
            for facial_landmarks in result.multi_face_landmarks:
                for i, landmark in enumerate(facial_landmarks.landmark):
                    # Extract x, y, z coordinates
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    # Write to the CSV file
                    writer.writerow([frame_index, i, x, y, z])

        frame_index += 1

    cap.release()

print("Landmark data saved to landmarks_data.csv")
