import time

import cv2
from pathlib import Path
import csv
from generic_pipeline.face_detection import detect_faces
from generic_pipeline.utils import plot_frame_with_bboxes, frame_generator, initialize_frame_dict
from generic_pipeline.hsemotion_model import HSEmotionModel


def get_face(bbox, frame):
    x, y, w, h, confidence_score = bbox
    face_img = frame[y:y + h, x:x + w]
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return face_img_rgb, confidence_score


def process_frame(frame):
    ret = initialize_frame_dict(emotion_classes)

    bboxes = detect_faces(frame)

    num_faces = len(bboxes)
    if num_faces == 0 or num_faces > 1:
        ret['num_faces'] = num_faces

    else:
        face_img_rgb, confidence_score = get_face(bboxes[0], frame)
        prediction = emotion_predictor.predict_emotions(face_img_rgb)
        ret['num_faces'] = num_faces
        ret['success'] = 1
        ret['confidence_score'] = confidence_score
        ret.update(prediction)

    return ret


def process_video(video_path: Path, sr: int, output_csv: Path):
    """
    Processes a video frame-by-frame to detect faces and predict emotions.
    """
    video_capture = cv2.VideoCapture(str(video_path))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        temp_frame_dict = initialize_frame_dict(emotion_classes)
        writer.writerow(temp_frame_dict.keys())

        for frame_idx, frame in frame_generator(video_capture, sr):
            if frame_idx % 1000 == 0:
                print(f"Processing frame {frame_idx} out of {total_frames}")

            try:
                frame_dict = process_frame(frame)
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue

            frame_dict['frame'] = frame_idx
            frame_dict['timestamp'] = frame_idx / fps

            writer.writerow(frame_dict.values())

    video_capture.release()


def process_dir(input_dir: Path, output_dir: Path, sr):
    # Iterate over all video files in the directory
    for idx, video_file in enumerate(input_dir.glob('*.mp4')):
        filename = video_file.stem
        print(f"summary - Processing video: {filename}")

        output_csv = Path(f'{output_dir}/{filename}.csv')
        # Process the video
        start = time.time()
        process_video(video_file, sr, output_csv)
        stop = time.time()
        processing_time_minutes = (stop - start) / 60
        print(f"summary - Done processing video: {filename}")
        print(f"summary - Processing took {processing_time_minutes:.2f} minutes.")


def process_single_file(filepath: Path, output_dir: Path, sr):
    filename = filepath.stem
    print(f"Processing video: {filename}")

    output_csv = Path(f'../out/predictions/{filename}.csv')
    # Process the video
    process_video(filepath, sr, output_csv)


emotion_predictor = HSEmotionModel(model_name="enet_b2_8_best")
emotion_classes = emotion_predictor.emotion_classes.values()

if __name__ == '__main__':
    sampling_rate = 5

    video_dir = Path('/media/user/TIMS-DISK/kosmos/split')
    out_dir = Path("/media/user/TIMS-DISK/kosmos/out/hsemotion_enet_b2_8_best_mediapipe_preds_better_output")
    process_dir(video_dir, out_dir, sampling_rate)

    # video_dir = Path('/home/tim/.sensitive_data/kosmos/split')
    # out_dir = Path("../out/predictions/kosmos_hsemotion_mediapipe")
    # process_dir(video_dir, out_dir, sampling_rate)

    # video_dir = Path("../data/videos/sentimotion")
    # out_dir = Path("../out/predictions/sentimotion_hsemotion_mediapipe")
    # process_dir(video_dir, out_dir, sampling_rate)

    # file_path = Path('/media/user/TIMS-DISK/kosmos/split/KOSMOS027_GK_BAS_LEFT.mp4')
    # out_dir = Path("../out/predictions/test")
    # process_single_file(file_path, out_dir, sampling_rate)
