import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
csv_file = '../out/predictions/audio_arousal_saustage/results.csv'
df = pd.read_csv(csv_file)

# Create a pivot table to count occurrences of each arousal label for each combination of emotion and intensity level
pivot_table = df.pivot_table(index=['emotion', 'intensity_level'], columns='arousal_label', aggfunc='size', fill_value=0)

# Plotting
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Arousal Label')
plt.ylabel('Emotion and Intensity Level')
plt.title('Arousal Label Distribution per Emotion and Intensity Level')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
