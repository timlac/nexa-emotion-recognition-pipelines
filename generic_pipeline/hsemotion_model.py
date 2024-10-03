import torch.nn.functional as F
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer

class HSEmotionModel:
    emotion_classes = {
        0: 'Anger',
        1: 'Contempt',
        2: 'Disgust',
        3: 'Fear',
        4: 'Happiness',
        5: 'Neutral',
        6: 'Sadness',
        7: 'Surprise'
    }

    def __init__(self, model_name: str = 'enet_b0_8_va_mtl', device: str = 'cuda'):
        self.device = device
        self.model_name = model_name
        self.fer = self.load_hsemotion_model()

    def load_hsemotion_model(self):
        """
        Loads the emotion recognition model.
        """
        return HSEmotionRecognizer(model_name=self.model_name, device=self.device)

    def predict_emotions(self, img):
        """
        Predicts emotions, valence, and arousal.
        """
        results = self.fer.predict_emotions(img, logits=True)
        logits = torch.tensor(results[1])

        # Apply softmax to the logits for emotions (first 8 values)
        emotion_logits = logits[:8]
        emotion_probabilities = F.softmax(emotion_logits, dim=0)

        # Create a dictionary mapping class names to their respective probabilities
        emotion_dict = {self.emotion_classes[i]: float(prob) for i, prob in enumerate(emotion_probabilities)}

        predicted_emotion_class = self.emotion_classes[torch.argmax(emotion_probabilities).item()]
        valence = logits[8].item()
        arousal = logits[9].item()

        ret = {
            'emotion_prob_dict': emotion_dict,
            'predicted_emotion': predicted_emotion_class,
            'valence': valence,
            'arousal': arousal
        }
        return ret