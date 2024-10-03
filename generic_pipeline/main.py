import cv2
from pathlib import Path
from emonet_utils.predict import EmotionPredictor
import csv
from generic_pipeline.face_detection import detect_faces
from generic_pipeline.utils import plot_frame_with_bboxes
from generic_pipeline.hsemotion_model import HSEmotionModel


def frame_generator(video_capture, sr):
    """
    :param video_capture: cv2.VideoCapture object
    :param sr: sampling rate
    :return: generator of frames
    """
    frame_idx = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_idx % sr == 0:
            yield frame_idx, frame
        frame_idx += 1


def process_video(video_path: Path, sampling_rate: int, output_csv: Path):
    """
    Processes a video frame-by-frame to detect faces and predict emotions.
    """
    filename = video_path.stem

    video_capture = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Frame', 'Timestamp', 'Face_ID', 'Confidence_Score', 'Predicted_Emotion', 'Valence',
                  'Arousal'] + list(emotion_predictor.emotion_classes.values())
        writer.writerow(header)

        for frame_idx, frame in frame_generator(video_capture, sampling_rate):

            bboxes = detect_faces(frame)
            timestamp = frame_idx / fps

            print(f"Detected {len(bboxes)} faces in frame {frame_idx}")

            # Sort bounding boxes by x-coordinate (left to right)
            bboxes.sort(key=lambda item: item[0])

            if len(bboxes) != 1:
                print(
                    f"detected {len(bboxes)} faces in video {filename}: {len(bboxes)} for frame {frame_idx}... skipping")
                plot_frame_with_bboxes(frame, bboxes, save_path=f'../out/frame_data/{filename}_frame_{frame_idx}.png')
                continue

            for face_id, bbox in enumerate(bboxes):

                try:
                    x, y, w, h, confidence_score = bbox
                    face_img = frame[y:y + h, x:x + w]

                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    # res = emotion_predictor.predict_emotion_from_array(face_img_rgb)

                    res = emotion_predictor.predict_emotions(face_img_rgb)

                    row = [frame_idx, timestamp, face_id, confidence_score, res['predicted_emotion'], res['valence'],
                           res['arousal']] + list(res['emotion_prob_dict'].values())
                    writer.writerow(row)

                    # plot_face(face_img, frame_idx, face_id)
                except Exception as e:
                    print(f"Error processing frame {frame_idx}, face {face_id}: {e}")
                    continue

    video_capture.release()


# device = 'cuda:0'
# emotion_predictor = EmotionPredictor(device)

emotion_predictor = HSEmotionModel()

def process_dir(input_dir: Path, output_dir: Path, sr):
    # Iterate over all video files in the directory
    for idx, video_file in enumerate(input_dir.glob('*.mp4')):
        if idx == 9:
            continue

        filename = video_file.stem
        print(f"Processing video: {filename}")

        output_csv = Path(f'{output_dir}/{filename}.csv')
        # Process the video
        process_video(video_file, sr, output_csv)


def process_sinlge_file(filepath: Path, output_dir: Path, sr):
    filename = filepath.stem
    print(f"Processing video: {filename}")

    output_csv = Path(f'../out/emotion_predictions_{filename}.csv')
    # Process the video
    process_video(filepath, sr, output_csv)



if __name__ == '__main__':
    sampling_rate = 5

    # video_dir = Path('/home/tim/.sensitive_data/kosmos/split')
    # process_dir(video_dir)

    video_dir = Path("../data/videos/sentimotion")
    out_dir = Path("../out/predictions/sentimotion_hsemotion_mediapipe")
    process_dir(video_dir, out_dir, sampling_rate)

    # file_path = Path('/home/tim/.sensitive_data/kosmos/split/KOSMOS021_RMW_LSI_LEFT.mp4')
    # process_sinlge_file(file_path)