import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = '../out/emotion_predictions_with_probabilities.csv'
df = pd.read_csv(csv_file)

# Plot 1: Valence and Arousal over Time
plt.figure(figsize=(10, 6))

# Plot Valence
plt.plot(df['Frame'], df['Valence'], label='Valence', color='blue')

# Plot Arousal
plt.plot(df['Frame'], df['Arousal'], label='Arousal', color='green')

# Customize the plot
plt.title('Valence and Arousal Over Time')
plt.xlabel('Frame')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# Plot 2: Predicted Emotion Probabilities over Time
plt.figure(figsize=(12, 8))

# Define the emotion columns
emotion_classes = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Plot each emotion's probability over time
for emotion in emotion_classes:
    plt.plot(df['Frame'], df[emotion], label=emotion)

# Customize the plot
plt.title('Emotion Probabilities Over Time')
plt.xlabel('Frame')
plt.ylabel('Probability')
plt.legend(loc='upper right')
plt.grid(True)

# Display the plot
plt.show()
