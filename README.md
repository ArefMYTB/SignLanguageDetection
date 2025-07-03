# Sign Language Action Recognition using LSTM (PyTorch & Keras)

This project implements **action recognition for sign language** using keypoint data extracted from video samples of the **WLASL (Word-Level American Sign Language)** dataset.

It includes:
- Keypoint extraction using **MediaPipe**
- A **custom LSTM model in PyTorch** (from scratch with PyTorch Lightning)
- A **baseline Keras model** using built-in `LSTM` layers
- Evaluation and confusion matrix

---

## Dataset: WLASL

The **WLASL (Word-Level American Sign Language)** video dataset.  
Each sample corresponds to a signer performing a specific sign word.

- Source: [WLASL GitHub](https://github.com/dxli94/WLASL)
- Number of classes (used in this subset): `4`  
  Example classes:
  - `book`
  - `drink`
  - `hello`
  - `idea`

---

## Keypoint Extraction with MediaPipe

We use **Google's MediaPipe** to extract body, hand, and face landmarks from each frame of video clips.

- Each video is converted into a sequence of frames.
- Keypoints per frame are concatenated into a single feature vector.
- Output shape per sample: `(num_frames, 1662)`  
  (`1662` = number of total keypoint features across hands, pose, and face)

All keypoints are stored in `.npy` files, grouped by class in subfolders under `keypoints/`.

---

## Model Implementations

### Keras Model

- Implemented with `tf.keras.Sequential`
- Uses 3 built-in `LSTM` layers + dense layers
- Padding/truncation applied using `pad_sequences`

### PyTorch Model (Custom LSTM)

- Built entirely from scratch with manual LSTM cell implementation
- Trained using PyTorch Lightning
- Sequence padding handled directly with tensor operations
- Xavier uniform initialization for weights
- Matched architecture and parameter count to the Keras model

Both models have the same total trainable parameters (596K) and got same accuracy of 86%.


## Training & Evaluation
### Training Details

- Both models trained for up to 2000 epochs
- Optimizer: Adam
- Loss function:
  - categorical_crossentropy (Keras)
  - CrossEntropyLoss (PyTorch)

