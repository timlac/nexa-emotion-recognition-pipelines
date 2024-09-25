from emonet_utils.predict import EmotionPredictor
from emonet_utils.utils import load_raw_image

# Parameters
device = 'cuda:0'
image_path = "../data/cropped_images/example.png"

# Initialize EmotionPredictor
emotion_predictor = EmotionPredictor(device)

# Load image
image_tensor = load_raw_image(image_path, device)

# Predict emotion
res = emotion_predictor.predict_emotion(image_tensor)
print(f"Predicted Emotion: {res['predicted_emotion']} - Valence: {res['valence']:.3f} - Arousal: {res['arousal']:.3f}")
print(f"Emotion Probabilities: {res['emotion_prob_dict']}")