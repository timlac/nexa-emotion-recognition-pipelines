import cv2

# Paths to your two videos
video_path_1 = '../data/videos/sentimotion/A65_disg_p_2.mov'
video_path_2 = '../data/videos/sentimotion/A65_ang_p_2.mov'

# Open the video files
cap1 = cv2.VideoCapture(video_path_1)
cap2 = cv2.VideoCapture(video_path_2)

# Get properties for both videos
fps1 = cap1.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
output_width = width1 + width2
output_height = max(height1, height2)
output_fps = min(fps1, fps2)

output_path = 'side_by_side_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Stop the loop if either video ends
    if not ret1 or not ret2:
        break

    # Resize frames to match height if necessary
    if height1 != height2:
        frame1 = cv2.resize(frame1, (width1, output_height))
        frame2 = cv2.resize(frame2, (width2, output_height))

    # Concatenate frames side by side
    combined_frame = cv2.hconcat([frame1, frame2])

    # Write the combined frame to the output video
    out.write(combined_frame)

# Release all resources
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_path}")
