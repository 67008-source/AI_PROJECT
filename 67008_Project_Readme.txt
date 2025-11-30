Facial Emotion Recognition (FER-2013) Deep Learning Project – Muhammad
Junaid Khan

Overview This project uses the FER-2013 dataset to train a deep learning
model for facial emotion recognition. The model classifies facial
expressions into 7 emotion categories using a CNN built with
TensorFlow/Keras.

This repository includes: - Dataset description - Preprocessing
pipeline - Model architecture - Training and validation - Performance
metrics - Saved model + history - Visualization & results

Dataset: FER-2013 The FER-2013 dataset contains 35,887 grayscale images,
all sized 48×48 pixels.

Emotion Classes: 0 - Angry 1 - Disgust 2 - Fear 3 - Happy 4 - Sad 5 -
Surprise 6 - Neutral

Dataset split: - Train: 28,709 images - Public Test: 3,589 images -
Private Test: 3,589 images

Images can be loaded through: - fer2013.csv (pixels stored as strings) -
dataset folder structure

Technologies Used: - Python 3 - Google Colab - TensorFlow / Keras -
NumPy, Pandas, Matplotlib, Seaborn - OpenCV (optional for face
detection)

Model Architecture: The model is a custom CNN designed for FER-2013: -
Conv2D layers with ReLU - MaxPooling2D - Dropout - Dense layers -
Softmax output (7 classes)

Designed for: - High accuracy - Low overfitting - Efficient training on
Colab

Training & Results: Training includes: - Normalization - Data
augmentation - Early stopping - Model checkpointing

Saved files: - fer2013_model.h5 - training_history.json -
model_performance.json

Retraining: from tensorflow.keras.models import load_model model =
load_model(‘/content/drive/MyDrive/fer2013_model.h5’)

Visualizations include: - Emotion distribution - Sample images -
Augmented data - Accuracy/loss graphs - Confusion matrix

Files Included: - fer2013_model.h5 - training_history.json -
model_performance.json - FER_Dataset_Description_Complete.ipynb - Main
training notebook

How to Run: 1. Upload notebooks to Colab 2. Mount Google Drive 3. Load
dataset/model 4. Run preprocessing 5. Train/retrain 6. Evaluate

Conclusion: This project builds an accurate CNN for emotion recognition
and can be extended to: - Real-time detection - Mobile app integration -
Raspberry Pi deployment - Advanced deep learning models

Author: Muhammad Junaid Khan AI Semester Project – 2025
