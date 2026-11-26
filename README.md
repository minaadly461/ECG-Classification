# ECG Classification using Deep Learning

## Project Overview

This project focuses on automated multi-label classification of 12-lead Electrocardiogram (ECG) signals using deep learning.

The goal is to identify five major cardiac diagnostic categories from ECG recordings using a CNN-LSTM architecture, while exploring different training strategies to improve classification performance.

## Dataset

**Dataset:** [PTB-XL – PhysioNet](https://physionet.org/content/ptb-xl/)

PTB-XL is a large publicly available ECG dataset containing:

- 21,799 clinical 12-lead ECG recordings.
- 10-second recordings sampled at 100 Hz or 500 Hz.
- 18,885 patients.
- Multiple diagnostic statements and annotations.

The project focuses on five diagnostic superclasses:

| Label | Description |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial Infarction |
| STTC | ST/T Change |
| CD | Conduction Disturbance |
| HYP | Hypertrophy |

The task is formulated as a **multi-label classification problem**, since an ECG recording may belong to more than one diagnostic category.

## Model Architecture

The main model is a **CNN-LSTM** architecture:

- **CNN layers:** Extract local patterns and morphological features from ECG signals.
- **LSTM layers:** Capture temporal dependencies across the signal.
- **Output layer:** Predict probabilities for the five diagnostic categories.

The model is trained using Binary Cross-Entropy loss and the Adam optimizer.

## Experiments

Several training and evaluation strategies were explored:

- Standard CNN-LSTM training.
- Class-weighted training to address class imbalance.
- Focal Loss to focus learning on difficult examples.
- Ensemble experiments combining model predictions.
- Cross-validation for more robust performance evaluation.
- Error analysis to investigate model mistakes.

## Evaluation

Model performance is evaluated using:

- ROC-AUC
- Precision
- Recall
- F1-Score
- Optimized classification thresholds

Individual decision thresholds are optimized for each diagnostic category using validation data.

## Experiment Tracking

**MLflow** is used to track training experiments, including:

- Model architecture and hyperparameters.
- Training configuration.
- Validation loss.
- Validation AUC.
- Saved model artifacts.

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Scikit-learn**
- **MLflow**
- **Jupyter Notebook**

## Project Structure

```text
ECG-Classification/
│
├── notebooks/
│   └── ECG_Classification.ipynb
│
├── src/
│   └── model.py
│
├── train.py
├── train_focal.py
├── train_weighted.py
│
├── evaluate.py
├── evaluate_focal.py
├── evaluate_weighted.py
├── evaluate_ensemble.py
├── evaluate_ensemble_focal.py
│
├── cross_validation.py
├── evaluate_cv.py
├── error_analysis.py
├── test.py
│
└── .gitignore
```

## Disclaimer

This project is developed for educational and research purposes. It is not intended to replace professional medical diagnosis or clinical decision-making.
