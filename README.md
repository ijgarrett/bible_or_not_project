# Bible-or-Not Text Classification

This project uses a dataset of Bible verses and non-Bible text to build a Random Forest Classifier in scikit-learn that classifies text as either **Bible** or **Not Bible**. I completed this project in August 2025, implementing a full pipeline for text preprocessing, feature extraction, model training, evaluation, and prediction.

## Project Overview

The goal is to predict whether a given piece of text comes from the Bible or not. The model uses TF-IDF vectorization to convert text into numerical features and a Random Forest Classifier for prediction. The trained model can classify both dataset test samples and new custom inputs.

## Dataset

- Contains labeled samples of Bible verses and non-Bible text
- Split:
  - ~62,000 samples for training
  - ~15,000 samples for testing
- Classes: `Bible`, `Not Bible`

## Tools and Libraries

- Python  
- scikit-learn (`RandomForestClassifier`, `TfidfVectorizer`, `StandardScaler`)  
- pandas & NumPy (data handling)  
- joblib (model persistence)  

## Process and Methodology

### 1. Data Preprocessing
- Collected Bible and non-Bible text samples
- Cleaned text (lowercasing, removing punctuation, tokenization)
- Converted text to numerical features using `TfidfVectorizer`

### 2. Feature Scaling
- Applied `StandardScaler` to normalize feature distributions  
- Saved the scaler as `scaler.joblib` for use during inference  

### 3. Model Training
- Trained a `RandomForestClassifier` with:
  - `n_estimators = 200`
  - `max_depth = None`
  - `n_jobs = -1` (parallelized across all CPU cores)
- Used GridSearchCV to tune hyperparameters  

### 4. Evaluation
- Evaluated accuracy on the test set
- Checked precision/recall and confusion matrix
- Tested predictions on custom inputs  

## Final Model Performance

- Test Set Accuracy: ~92%  
- Model demonstrates strong generalization to unseen text  

## Files in This Project

- `train_model.py`: script for training the Random Forest model  
- `predict.py`: script for making predictions with the trained model  
- `bible_or_not_model.joblib`: saved trained model  
- `scaler.joblib`: saved StandardScaler for preprocessing  
- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`: feature and label arrays (ignored in `.gitignore` due to size)  
- `README.md`: summary of the project, purpose, and methods  

## Timeline

Project completed in late August 2025.  

## Future Improvements

- Try deep learning methods (e.g., LSTMs, Transformers like BERT)  
- Expand dataset with more diverse non-Bible text for robustness  
- Perform more extensive hyperparameter optimization  
- Add cross-validation and ensemble methods for higher accuracy  
- Deploy model as a web app for real-time classification  

---
