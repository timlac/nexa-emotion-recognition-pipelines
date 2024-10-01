import cv2
import ffmpeg
import os

# SLOOOOOOOW, use ffmpeg instead

# Define input and output directories
input_dir = '/home/tim/.sensitive_data/kosmos/original'
output_dir = '/home/tim/.sensitive_data/kosmos/split'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all video files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".mp4"):
        print("Processing", filename)

        input_video_path = os.path.join(input_dir, filename)

        # Define output file paths for left and right halves
        base_name = os.path.splitext(filename)[0]
        left_output_video_path = os.path.join(output_dir, f"{base_name}_LEFT.mp4")
        # right_output_video_path = os.path.join(output_dir, f"{base_name}_RIGHT.mp4")

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 720

        # Define codec and create VideoWriter objects for left and right halves
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_left = cv2.VideoWriter(left_output_video_path, fourcc, fps, (width // 2, height))  # 640x720
        # out_right = cv2.VideoWriter(right_output_video_path, fourcc, fps, (width // 2, height))  # 640x720

        # Loop through the video and split each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Crop the left half (640x720)
            left_half = frame[:, :640]

            # Crop the right half (640x720)
            right_half = frame[:, 640:]

            # Write the frames to the output video files
            out_left.write(left_half)

        # Release everything
        cap.release()
        out_left.release()

        # Merge audio using ffmpeg (copy the audio stream without re-encoding)
        ffmpeg.input(input_video_path).output(left_output_video_path, c='copy', an=None).run()

        print(f"Processed {filename} into {base_name}_LEFT.mp4 and {base_name}_RIGHT.mp4")

print("All videos processed.")
