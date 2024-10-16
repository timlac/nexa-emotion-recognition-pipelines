import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


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


def plot_frame_with_bboxes(frame, bboxes, save_path=None):
    """
    Plots the frame with bounding boxes around detected faces and confidence scores.
    """
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax = plt.gca()

    for bbox in bboxes:
        x, y, w, h, confidence_score = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 10, f'{confidence_score:.2f}', color='red', fontsize=8, weight='bold')

    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def plot_face(face_img, frame_idx, face_id):
    """
    Plots the detected face image.
    """
    plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Frame: {frame_idx}, Face ID: {face_id}')
    plt.axis('off')
    plt.show()