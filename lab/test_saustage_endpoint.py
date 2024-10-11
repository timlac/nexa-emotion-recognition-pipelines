import requests
import json
from pathlib import Path
import pandas as pd
from nexa_sentimotion_filename_parser.metadata import Metadata
from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper

Mapper._load_data_if_needed()

# Define the URL of the API endpoint
url = 'http://localhost:5000/infer_arousal'

# Define the directory containing the video files
video_dir = Path('/home/tim/Work/nexa/nexa-emotion-recognition-pipelines/data/videos/sentimotion')

# List to store the results
results = []

# Iterate over all video files in the directory
for file_path in video_dir.glob('*.mp4'):
    filename = file_path.stem
    meta = Metadata(filename)
    emotion = Mapper.get_emotion_from_id(meta.emotion_1_id)
    intensity_level = meta.intensity_level if meta.intensity_level is not None else 2

    # Create the payload
    payload = {
        'file_path': str(file_path)
    }

    # Send the POST request to the API endpoint
    response = requests.post(url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        print(result)
        result.update({
            'filename': filename,
            'emotion': emotion,
            'intensity_level': intensity_level
        })
        results.append(result)
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Convert the results list to a pandas DataFrame
df = pd.DataFrame(results)

# Save the DataFrame as a CSV file
output_csv = Path('../out/predictions/audio_arousal_saustage/results.csv')
df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")