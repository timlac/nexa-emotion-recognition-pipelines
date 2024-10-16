from generic_pipeline.utils import frame_generator
from generic_pipeline.face_detection import detect_faces


from pathlib import Path
import cv2


def my_func(path, img_path):
    video_capture = cv2.VideoCapture(str(path))

    for frame_idx, frame in frame_generator(video_capture, 20):
        print(f"Frame {frame_idx}")

        bboxes = detect_faces(frame)
        for i, (x, y, w, h, confidence_score) in enumerate(bboxes):
            face_img = frame[y:y + h, x:x + w]
            face_img_path = f"{img_path}/face_{frame_idx}_{i}.png"
            cv2.imwrite(face_img_path, face_img)
            print(f"Saved face {i} from frame {frame_idx} to {face_img_path}")


if __name__ == '__main__':
    filepath = Path("/home/tim/.sensitive_data/kosmos/original_without_left_right_division/KOSMOS117_TH_BAS.mp4")
    face_out_path = "../out/frame_data/temp_frames"

    my_func(filepath, face_out_path)


