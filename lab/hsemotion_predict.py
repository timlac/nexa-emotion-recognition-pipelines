from hsemotion.facial_emotions import HSEmotionRecognizer
import torch.nn.functional as F
import torch
import numpy as np


hs_emotion_classes = ['Anger',
                      'Contempt',
                      'Disgust',
                      'Fear',
                      'Happiness',
                      'Neutral',
                      'Sadness',
                      'Surprise',
                      ]


def load_hsemotion_model(model_name: str, device: str):
    """
    Loads the emotion recognition model.
    """
    # Loading the model
    fer = HSEmotionRecognizer(model_name=model_name, device=device)
    return fer

def predict_emotions(fer: HSEmotionRecognizer, img):
    """
    Predicts emotions, valence, and arousal.
    """
    results = fer.predict_emotions(img, logits=True)

    logits = torch.tensor(results[1])

    # Apply softmax to the logits for emotions (first 8 values)
    emotion_logits = logits[:8]
    emotion_probabilities = F.softmax(emotion_logits, dim=0)

    # Create a dictionary mapping class names to their respective probabilities
    emotion_dict = {emotion: float(prob) for emotion, prob in zip(hs_emotion_classes, emotion_probabilities)}

    predicted_emotion_class = torch.argmax(emotion_probabilities).item()

    valence = logits[8].item()
    arousal = logits[9].item()

    ret = {
        'emotion_prob_dict': emotion_dict,
        'predicted_emotion': predicted_emotion_class,
        'valence': valence,
        'arousal': arousal
    }
    return ret