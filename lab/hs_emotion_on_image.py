from hsemotion.facial_emotions import HSEmotionRecognizer
import cv2

# Read the image
img = cv2.imread('../data/fer2013/train/angry/Training_7928172.jpg')

class2index = {'Anger': 0, 'Contempt': 1, 'Disgust': 2, 'Fear': 3, 'Happiness': 4,
               'Neutral': 5, 'Sadness': 6, 'Surprise': 7, 'Valence': 8, 'Arousal': 9}

index2class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5:
    'Neutral', 6: 'Sadness', 7: 'Surprise', 8: 'Valence', 9: 'Arousal'}

# Use a model that supports valence-arousal prediction
model_name = 'enet_b0_8_va_mtl'
fer = HSEmotionRecognizer(model_name=model_name, device='cuda')  # Use 'cpu' if CUDA is not available

# Predict emotions, valence, and arousal
results = fer.predict_emotions(img, logits=True)

dominant_emotion = results[0]
result_array = results[1]

# Create a dictionary with emotions and their corresponding probabilities
emotion_probs = {index2class[i]: prob.item() for i, prob in enumerate(result_array)}

# Output the probabilities for each emotion
print(emotion_probs)

print(f'Dominant emotion: {dominant_emotion}')
print(f'Result array: {result_array}')
print(f'Emotion: {emotion_probs}')