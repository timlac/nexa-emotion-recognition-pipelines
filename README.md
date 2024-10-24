# Emotion Recognition

### TODO

- **Look into using the face with the highest confidence score in the generic pipeline if multiple faces are detected.**
- Evaluate if a different method, e.g. face-alignment, is better for face detection for some videos that are tricky.
- **Implement openface style output files:** 
  - Write all frames to csv regardless if face is captured or not, set success to 0 if no face is detected.
  - Potentially get rid of the strict confidence threshold
  - If there are multiple faces detected, either choose the one with the highest confidence score or simply write both to the csv (as face 0, face 1, etc.)

## Models

### Face Models

[FER](https://github.com/JustinShenk/fer) is a CNN based on the FER-2013 dataset.

[VGG-Face](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) also a CNN based on FER-2013 dataset.

[Emonet](https://github.com/face-analysis/emonet) Official implementation of the paper "Estimation of continuous 
valence and arousal levels from faces in naturalistic conditions". Pretrained models are available from AffectNet dataset. 
After doing some validation experiments on the Emonet model, the results are quite counterintuitive both on the 
FER-2013 and Sentimotion datasets. It seems [other people](https://github.com/face-analysis/emonet/issues/18) have had trouble
validating the model for AffectNet as well. This could be an issue with how the images are pre-processed, or some
other issue, needs further investigation.

[HSEmotion](https://github.com/av-savchenko/face-emotion-recognition) Multipurpose library. Valence and arousal estimation
can be achieved using [this model](https://github.com/av-savchenko/face-emotion-recognition/issues/24). Video [demo](https://github.com/av-savchenko/hsemotion-onnx/blob/main/demo/recognize_emotions_video.py). It's important to have the right version of timm library, 
see [github comment](https://github.com/av-savchenko/hsemotion/issues/4#issuecomment-1722394042). Seems to be trained on VGG-Face2 dataset.

[Facetorch](https://github.com/tomas-gajarsky/facetorch) is a Python library that can detect faces and analyze facial features using deep neural networks.
It gathers open sourced face analysis tools from various sources. Utilizes HSEmotion (above) for emotion recognition and [ELIM](https://github.com/kdhht2334/ELIM_FER)
for valence/arousal estimation. Currently waiting for a reply on [this issue](https://github.com/tomas-gajarsky/facetorch/issues/78). 

### Speech Models

[speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) wav2vec based
easy model trained on IEMOCAP dataset.

[r-f/wav2vec-english-speech-emotion-recognition](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition) also wav2vec based. 
Fine-tuned on several datasets. 
- Surrey Audio-Visual Expressed Emotion (SAVEE) - 480 audio files from 4 male actors
- Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) - 1440 audio files from 24 professional actors (12 female, 12 male)
- Toronto emotional speech set (TESS) - 2800 audio files from 2 female actors

Seems to be based on an english wav2vec model which is a downside. On the other hand the datasets seem english as well so might not be a problem. 

[SenseVoice](https://github.com/FunAudioLLM/SenseVoice) is a speech foundation model with multiple speech understanding capabilities, 
including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and audio event detection (AED).


[sustAge](https://github.com/EIHW/sustAGE_ArousalRecognition) Arousal recognition toolkit. 

## Datasets 

[VGGFace2](https://github.com/ox-vgg/vgg_face2) The dataset contains 3.31 million images of 9131 subjects (identities), 
with an average of 362.6 images for each subject. Images are downloaded from Google Image Search and have large variations in pose, age, 
illumination, ethnicity and profession (e.g. actors, athletes, politicians). 

[Hume Vocal Burst Database (H-VB).](https://zenodo.org/records/6320973) referenced [here](https://ieeexplore.ieee.org/abstract/document/10095294). 

[FER-2013](https://paperswithcode.com/dataset/fer2013) contains approximately **30,000** facial RGB images of different
expressions with size restricted to 48×48, and the main labels of it can be divided into **7 types:
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral**.
The Disgust expression has the minimal number of images – 600, while other labels have nearly 5,000 samples each.

[AffectNet](https://paperswithcode.com/dataset/affectnet) is a large facial expression dataset with around 
**0.4 million** images manually labeled for the presence of **8 (neutral, happy, angry, sad, fear, surprise, disgust, contempt) 
facial expressions along with the intensity of valence and arousal**. Leaderboard for models here:
[AffectNet Leaderboard](https://paperswithcode.com/sota/emotion-recognition-on-affectnet).

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

## Considerations

### Face Detection

The expected bounding box may vary depending on the FER model used. Emonet uses [face-alignment](https://github.com/1adrianb/face-alignment?tab=readme-ov-file) python lib.
[Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) also seems to used extensively. 

- Face detection is necessary to detect faces in images and videos.
- Face tracking may be necessary to track faces in videos where there are multiple faces.
  - Face tracking which relies on creating new embeddings for each frame is computationally expensive.
  - Using some technology in order to not have to check the embeddings every frame (e.g. optical flow) is a better option.
- Need to keep track of how many faces are detected in the output format, potentially some logging system to identify when no faces are detected.

Currently looking into using vector indexes along with deepface to index all the faces in a video in the index 
and then query similar faces. Current problem is that I need to perform range search to do this. But potentially 
it would be sufficient to just retrieve all the items with their distances to some image and filter out the ones that are too far away.

However, this method might fail if the image we use as reference happens to be in a weird light or similar. 

It would be optimal if we could cluster all the vector in some way that picks the number of clusters dynamically.... 
Perhaps we could use Silhouette Score for this, however now we're getting into more complex solutions which might slow down
the computation time... 


### The state of emotion research

Hume AI has interesting models especially for speech emotion recognition.

Models that relies only on facial expression are somewhat limited since they mainly only rely on the facial expression, 
and does not take into account for example head pose. While some datasets include valence and arousal, e.g. AffectNet, 
there is no intensity estimation built in for specific emotions. Furthermore, the number of emotions is quite limited. 

Perhaps it would make sense to try to train our own models for emotion recognition using our own dataset (which has 44 different emotions). 
This however is a complex task that requires both technical knowledge and adequate compute resources. 

### Segmentation of sound

In order to make speech emotion recognition more accurate, we need to be able to segment the audio according to speaker. 
SOTA models for speaker diarization, that I have seen so far does not seem to perform adequately.
Have tried Pyannote/speaker diarization so far, see repo nexa-transcription. 

### Specific Features of interest 

Gaze direction has been shown to be associated with "the underlying behavioral intent (approach-
avoidance) communicated by an emotional expression” [2]. Milders et al. [42] showed that the gaze
direction of another person can affect your emotion recognition accuracy and the intensity by
which you perceive the emotional stimuli. Moreover, results showed that averted gaze is a useful
feature when detecting fear over happiness or anger and the exact opposite with a direct gaze. In
addition, [34] indicated that when humans experience embarrassment, they first avert their gaze
and then subsequently additional expressions occur, such as shifting eye, abnormal speech sounds,
and smiling. (Snippet from this [paper](data/papers/Crossmodal Embeddings for Emotion Recognition.pdf))


### Validation 

Try validating emonet on sentimotion subset using face_alignment library for face detection instead of mediapipe. 

Results using mediapipe on sentimotion are very strange. 