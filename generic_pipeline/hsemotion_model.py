import torch.nn.functional as F
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer

class HSEmotionModel:
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

        if self.model_name == "enet_b0_8_va_mtl":
            valence = logits[8].item()
            arousal = logits[9].item()
        else:
            valence = 0
            arousal = 0

        ret = {
            'predicted_emotion': predicted_emotion_class,
            'valence': valence,
            'arousal': arousal
        }
        ret.update(emotion_dict)

        return ret


if __name__ == '__main__':
    model = HSEmotionModel()

    print(model.emotion_classes.values())