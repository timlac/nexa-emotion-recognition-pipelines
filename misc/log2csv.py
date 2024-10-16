import re
import pandas as pd

# Define the regex patterns to capture data
video_pattern = re.compile(
    r'SUMMARY: Summary for video (.+):')  # Adjusted to capture filenames with spaces and special characters
total_frames_pattern = re.compile(r'SUMMARY: Total frames: (\d+)')
faces_0_pattern = re.compile(r'SUMMARY: Frames with 0 faces: \d+ \(([\d.]+)%\)')
faces_1_pattern = re.compile(r'SUMMARY: Frames with 1 face: \d+ \(([\d.]+)%\)')
multiple_faces_pattern = re.compile(r'SUMMARY: Frames with multiple faces: \d+ \(([\d.]+)%\)')


log_file_path = '/media/user/TIMS-DISK/kosmos/out/hsemotion_mediapipe_logs.txt'
output_csv_path = 'summary_log.csv'

# Initialize a list to store the parsed data
parsed_data = []

# Read and parse the log file
with open(log_file_path, 'r') as file:
    video_data = {}
    for line in file:
        video_match = video_pattern.search(line)
        if video_match:
            if video_data:  # Save the previous video's data if it exists
                parsed_data.append(video_data)
            video_data = {'filename': video_match.group(1)}

        total_frames_match = total_frames_pattern.search(line)
        if total_frames_match:
            video_data['total_frames'] = int(total_frames_match.group(1))

        faces_0_match = faces_0_pattern.search(line)
        if faces_0_match:
            video_data['percentage_0_faces'] = float(faces_0_match.group(1))

        faces_1_match = faces_1_pattern.search(line)
        if faces_1_match:
            video_data['percentage_1_face'] = float(faces_1_match.group(1))

        multiple_faces_match = multiple_faces_pattern.search(line)
        if multiple_faces_match:
            video_data['percentage_multiple_faces'] = float(multiple_faces_match.group(1))

    if video_data:  # Don't forget the last video data
        parsed_data.append(video_data)

# Create a DataFrame using pandas
df = pd.DataFrame(parsed_data)

# Select the columns, now including total_frames
df = df[['filename', 'total_frames', 'percentage_0_faces', 'percentage_1_face', 'percentage_multiple_faces']]

# Write the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

print(f"CSV file created: {output_csv_path}")