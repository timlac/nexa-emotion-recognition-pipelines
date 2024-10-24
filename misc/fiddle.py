import pandas as pd


emotion_classes = {
    0: 'anger',
    1: 'contempt',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'neutral',
    6: 'sadness',
    7: 'surprise'
}

df = pd.read_csv("../out/predictions/KOSMOS027_GK_BAS_LEFT.csv")

df_success = df[df["success"] == 1]

print(df_success)

print(df[df.index == 55])