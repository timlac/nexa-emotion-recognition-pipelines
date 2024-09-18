# Emotion Recognition

## Models

[FER](https://github.com/JustinShenk/fer) is a CNN based on the FER-2013 dataset.

[VGG-Face](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) also a CNN based on FER-2013 dataset.

[Emonet](https://github.com/face-analysis/emonet) Official implementation of the paper "Estimation of continuous 
valence and arousal levels from faces in naturalistic conditions". Pretrained models are available from AffectNet dataset. 

## Datasets 

[FER-2013](https://paperswithcode.com/dataset/fer2013) contains approximately **30,000** facial RGB images of different
expressions with size restricted to 48×48, and the main labels of it can be divided into **7 types:
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral**.
The Disgust expression has the minimal number of images – 600, while other labels have nearly 5,000 samples each.

[AffectNet](https://paperswithcode.com/dataset/affectnet) is a large facial expression dataset with around 
**0.4 million** images manually labeled for the presence of **8 (neutral, happy, angry, sad, fear, surprise, disgust, contempt) 
facial expressions along with the intensity of valence and arousal**.

[RAVDESS](https://paperswithcode.com/dataset/ravdess) Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7,356 files (total size: 24.8 GB). 
The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements 
in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, 
and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of 
emotional intensity (normal, strong), with an additional neutral expression. 

[GEMEP](https://www.unige.ch/cisa/gemep) The GEneva Multimodal Emotion Portrayals (GEMEP) is a collection of audio and 
video recordings featuring 10 actors portraying 18 affective states, with different verbal contents and different modes of expression.  

## Helper libraries 

[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) can be used to quickle parse videos into frames for image based models. 
It has built in models for [Face Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector), 
[Pose Landmarks](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) and other tasks. 