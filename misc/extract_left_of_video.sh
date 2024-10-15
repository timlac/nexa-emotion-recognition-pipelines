#!/bin/bash

# Set your input and output directories
input_dir="/media/user/TIMS-DISK/kosmos/original"
output_dir="/media/user/TIMS-DISK/kosmos/split"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop over all video files in the input directory
for video in "$input_dir"/*.mp4; do
    # Extract the filename without the directory or extension
    filename=$(basename -- "$video")
    name="${filename%.*}"

    # Define the output file path
    output_file="$output_dir/${name}_LEFT.mp4"

    # Check if the output file already exists
    if [ -f "$output_file" ]; then
        echo "Skipping $filename as it has already been processed."
        continue
    fi

    # Split into left half
    ffmpeg -i "$video" -filter:v "crop=640:720:0:0" -c:v libx264 -preset ultrafast -c:a copy "$output_file"

    echo "Processed $filename"
done

echo "All videos processed and saved in $output_dir"
