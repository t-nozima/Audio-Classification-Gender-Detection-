#  Audio Classification - Gender Detection (Uzbek Common Voice)

This project demonstrates how to preprocess audio data and build a basic machine learning model that classifies the speaker’s gender (male or female) using the Common Voice 13.0 Uzbek dataset.

---

##  Objective

The goal is to:
- Load and clean raw audio data
- Extract useful features like MFCCs, chroma, and spectral contrast
- Train a simple machine learning model
- Evaluate its performance and share insights

This lays the foundation for building smarter audio-based systems in the future.

---

##  Dataset

- **Source:** [Mozilla Common Voice 13.0 - Uzbek](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)
- **Split Used:** Train + Validation
- **Labels:** Gender (`male`, `female`)

---

##  Tools & Libraries

- `librosa`, `soundfile` – Audio processing
- `pandas`, `numpy` – Data handling
- `scikit-learn` – Machine learning
- `datasets` (Hugging Face) – Loading dataset
- `tqdm` – Progress tracking

---

##  Steps Taken

### 1. **Data Loading**
- Used Hugging Face’s `datasets` to load the Uzbek subset of Common Voice.
- Filtered samples to only include entries labeled with gender: male or female.

### 2. **Audio Preprocessing & Feature Extraction**
Extracted the following features for each audio file using `librosa`:
- **MFCCs (Mel-Frequency Cepstral Coefficients)**
- **Chroma Features**
- **Spectral Contrast**

Clips shorter than 1 second were skipped. Feature vectors were concatenated for model input.

### 3. **Dataset Preparation**
- Converted the extracted features and labels to NumPy arrays.
- Encoded categorical labels using `LabelEncoder`.
- Split the dataset using an 80/20 train-test split, stratified by label.

### 4. **Modeling**
- Trained a baseline **Random Forest Classifier** (`n_estimators=100`).
- Evaluated the model on test data using accuracy and classification metrics.

### 5. **Saving Outputs**
- Final processed dataset was saved as `processed_audio_features.csv`.

---

##  Model Performance

**Final Accuracy:** `88%`

```text
              precision    recall  f1-score   support

      female       0.86      0.81      0.83      2447
        male       0.89      0.92      0.91      4246

    accuracy                           0.88      6693
   macro avg       0.87      0.86      0.87      6693
weighted avg       0.88      0.88      0.88      6693
