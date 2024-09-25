import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from emonet_utils import load_emonet, load_raw_image, predict_emotion, load_image_from_array

# Read the image
# Path to the FER-2013 test dataset
test_dir = '../data/fer2013/test'

predicted_label = []

# Parameters
device = 'cuda:0'
image_size = 256
image_path = "../data/cropped_images/example.png"

# Load model
net = load_emonet(device)

predictions = []

# Loop through the test dataset
for emotion_folder in os.listdir(test_dir):
    emotion_folder_path = os.path.join(test_dir, emotion_folder)

    # Check if it's a valid folder (emotion label)
    if os.path.isdir(emotion_folder_path):
        # Loop through images in the folder
        for idx, img_file in enumerate(os.listdir(emotion_folder_path)):
            if idx > 100:
                continue  # Limit the number of images to process

            # Predict emotions, valence, and arousal
            img_path = os.path.join(emotion_folder_path, img_file)

            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip invalid images

            image_tensor = load_image_from_array(img, image_size, device)

            emotion, valence, arousal = predict_emotion(net, image_tensor)

            print(f"actual emotion: {emotion_folder}, predicted emotion: {emotion}, valence: {valence:.3f}, arousal: {arousal:.3f}")

            predictions.append({
                'emotion': emotion_folder,
                'dominant_emotion': emotion,
                'valence': valence,
                'arousal': arousal
            })


# Convert predictions to a DataFrame for easy visualization
df = pd.DataFrame(predictions)

# Valence and Arousal Box Plot for Each Emotion
plt.figure(figsize=(10, 6))
sns.boxplot(x='emotion', y='valence', data=df)
plt.title('Valence Distribution per Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='emotion', y='arousal', data=df)
plt.title('Arousal Distribution per Emotion')
plt.show()

# Scatter plot of Valence vs Arousal colored by emotion
plt.figure(figsize=(10, 6))
sns.scatterplot(x='valence', y='arousal', hue='dominant_emotion', data=df, palette="deep")
plt.title('Valence vs Arousal Distribution')
plt.show()

# # Histogram of Predicted Dominant Emotions
# plt.figure(figsize=(10, 6))
# sns.countplot(x='dominant_emotion', data=df, order=df['dominant_emotion'].value_counts().index)
# plt.title('Distribution of Predicted Dominant Emotions')
# plt.show()


# Stacked Bar Plot: Distribution of True vs Predicted Emotions
plt.figure(figsize=(12, 6))
true_predicted_counts = pd.crosstab(df['emotion'], df['dominant_emotion'])
true_predicted_counts.div(true_predicted_counts.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, colormap="tab20", figsize=(12, 6))

plt.title('True vs Predicted Emotion Distribution (Stacked)')
plt.ylabel('Proportion of Predicted Emotions')
plt.xlabel('True Emotion')
plt.legend(title='Predicted Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Confusion Matrix Heatmap: True vs Predicted Emotions
plt.figure(figsize=(10, 8))
conf_matrix = pd.crosstab(df['emotion'], df['dominant_emotion'])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title('Confusion Matrix: True vs Predicted Emotions')
plt.ylabel('True Emotion')
plt.xlabel('Predicted Emotion')
plt.show()