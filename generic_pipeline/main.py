import cv2
from pathlib import Path
import csv
from generic_pipeline.face_detection import detect_faces
from generic_pipeline.utils import plot_frame_with_bboxes, frame_generator
from generic_pipeline.hsemotion_model import HSEmotionModel
from generic_pipeline.summary_logger import SummaryLogger


def process_video(video_path: Path, sr: int, output_csv: Path):
    """
    Processes a video frame-by-frame to detect faces and predict emotions.
    """
    filename = video_path.stem

    video_capture = cv2.VideoCapture(str(video_path))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    face_counts = {}

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Frame', 'Timestamp', 'Face_ID', 'Confidence_Score', 'Predicted_Emotion', 'Valence',
                  'Arousal'] + list(emotion_predictor.emotion_classes.values())
        writer.writerow(header)

        for frame_idx, frame in frame_generator(video_capture, sr):

            bboxes = detect_faces(frame)
            timestamp = frame_idx / fps

            num_faces = len(bboxes)
            face_counts[frame_idx] = num_faces

            print(f"Detected {len(bboxes)} faces in frame {frame_idx}")

            # Sort bounding boxes by x-coordinate (left to right)
            # bboxes.sort(key=lambda item: item[0])

            # TODO: As it currently stands, we discard frames with multiple faces.
            # Since most frames should only contain one face the second face is likely a false positive with low confidence,
            # thus, we could quite safely simply process the face with the highest confidence.
            # Sort bounding boxes by confidence score (highest to lowest)
            # bboxes.sort(key=lambda item: item[4], reverse=True)

            # TODO: only discard frames with 0 faces detected
            if len(bboxes) != 1:
                print(
                    f"detected {len(bboxes)} faces in video {filename}: {len(bboxes)} for frame {frame_idx}... skipping")
                # plot_frame_with_bboxes(frame, bboxes, save_path=f'../out/frame_data/{filename}_frame_{frame_idx}.png')
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
    print(f"SUMMARY: Summary for video {filename}:")
    summary_logger.log_summary(face_counts, filename)


def process_dir(input_dir: Path, output_dir: Path, sr):
    # Iterate over all video files in the directory
    for idx, video_file in enumerate(input_dir.glob('*.mp4')):
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


emotion_predictor = HSEmotionModel(model_name="enet_b2_8_best")
summary_logger = SummaryLogger('/media/user/TIMS-DISK/kosmos/out/hsemotion_enet_b2_8_best_mediapipe_preds_log_summary.csv')


if __name__ == '__main__':
    sampling_rate = 10

    video_dir = Path('/media/user/TIMS-DISK/kosmos/split')
    out_dir = Path("/media/user/TIMS-DISK/kosmos/out/hsemotion_enet_b2_8_best_mediapipe_preds")
    process_dir(video_dir, out_dir, sampling_rate)

    # video_dir = Path('/home/tim/.sensitive_data/kosmos/split')
    # out_dir = Path("../out/predictions/kosmos_hsemotion_mediapipe")
    # process_dir(video_dir, out_dir, sampling_rate)

    # video_dir = Path("../data/videos/sentimotion")
    # out_dir = Path("../out/predictions/sentimotion_hsemotion_mediapipe")
    # process_dir(video_dir, out_dir, sampling_rate)

    # file_path = Path('/home/tim/.sensitive_data/kosmos/split/KOSMOS021_RMW_LSI_LEFT.mp4')
    # process_sinlge_file(file_path)

