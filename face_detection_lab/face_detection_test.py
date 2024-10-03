import matplotlib.pyplot as plt
from face_alignment.detection.sfd.sfd_detector import SFDDetector
import cv2
import os
import face_alignment
from pathlib import Path

def process_image(img_path, model):
    # Load the image using cv2
    input_image = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if input_image is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")

    # Convert the image to RGB format
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    det = model.detect_from_image(input_image_rgb)

    if len(det) == 0:
        print("No faces detected.")
        # Use matplotlib to display the result
        plt.imshow(input_image_rgb)
        plt.axis('off')  # Hide axis for a cleaner display
        plt.title("Detected Faces")
        plt.show()  # Display the image

    # Loop through detected faces
    for face in det:
        x1, y1, x2, y2, confidence = face
        print(f"confidence: {confidence}")

        if confidence > 0.5:  # Filter based on confidence threshold
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Draw the rectangle (bounding box)
            input_image_rgb = cv2.rectangle(input_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert BGR (OpenCV default) to RGB for displaying in matplotlib
    input_rgb = cv2.cvtColor(input_image_rgb, cv2.COLOR_BGR2RGB)

    # Use matplotlib to display the result
    plt.imshow(input_rgb)
    plt.axis('off')  # Hide axis for a cleaner display
    plt.title("Detected Faces")
    plt.show()  # Display the image

def process_images_in_directory(directory_path):
    # Initialize the face detector
    model = face_alignment.detection.sfd.sfd_detector.SFDDetector(device="cuda")

    # Iterate over all image files in the directory
    for img_path in Path(directory_path).glob('*.png'):
        print(f"Processing image: {img_path}")
        process_image(str(img_path), model)

# Directory containing the images
directory_path = '../out/frame_data'

# Process all images in the directory
process_images_in_directory(directory_path)