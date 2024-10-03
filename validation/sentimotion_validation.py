from pathlib import Path
import os
import pandas as pd
from nexa_sentimotion_filename_parser.metadata import Metadata
from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper
import matplotlib.pyplot as plt
import seaborn as sns

Mapper._load_data_if_needed()

file_dir = Path('../out/predictions/sentimotion_emonet_mediapipe')

data_summary = []

for p in file_dir.glob('*.csv'):
    filename = Path(p).stem
    meta = Metadata(filename)

    df = pd.read_csv(p)
    mean_arousal = df["Arousal"].mean()
    mean_valence = df["Valence"].mean()
    dominant_emotion = df["Predicted_Emotion"].mode()[0]

    d = {"filename": meta.filename,

        "emotion": Mapper.get_emotion_from_id(meta.emotion_1_id),

         "intensity_level": meta.intensity_level if meta.intensity_level is not None else 2,
         "mean_arousal": mean_arousal, "mean_valence": mean_valence,
         "predicted_emotion": dominant_emotion}

    data_summary.append(d)

df = pd.DataFrame(data_summary)

# Create a box plot for valence grouped by emotion and intensity level
plt.figure(figsize=(20, 12))
sns.boxplot(x='emotion', y='mean_valence', data=df)
plt.xlabel('Emotion')
plt.ylabel('Mean Valence')
plt.title('Distribution of Mean Valence by Emotion and Intensity Level')
plt.legend(title='Intensity Level')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()

plt.show()

# Create a box plot for arousal grouped by emotion and intensity level
plt.figure(figsize=(20, 12))
sns.boxplot(x='emotion', y='mean_arousal', data=df)
plt.xlabel('Emotion')
plt.ylabel('Mean Arousal')
plt.title('Distribution of Mean Arousal by Emotion and Intensity Level')
plt.legend(title='Intensity Level')
plt.xticks(rotation=90)

plt.grid(True)
plt.tight_layout()

plt.show()

# Stacked Bar Plot: Distribution of True vs Predicted Emotions
plt.figure(figsize=(12, 6))
true_predicted_counts = pd.crosstab(df['emotion'], df['predicted_emotion'])
true_predicted_counts.div(true_predicted_counts.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, colormap="tab20", figsize=(12, 6))

plt.title('True vs Predicted Emotion Distribution (Stacked)')
plt.ylabel('Proportion of Predicted Emotions')
plt.xlabel('True Emotion')
plt.legend(title='Predicted Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Confusion Matrix Heatmap: True vs Predicted Emotions
plt.figure(figsize=(16, 8))
conf_matrix = pd.crosstab(df['emotion'], df['predicted_emotion'])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title('Confusion Matrix: True vs Predicted Emotions')
plt.ylabel('True Emotion')
plt.xlabel('Predicted Emotion')
plt.tight_layout()

plt.show()