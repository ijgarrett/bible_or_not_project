# Bible-or-Not Text Classification

This project uses a dataset of Bible verses and non-Bible text to build a Random Forest Classifier and a Logistic Regression Classifier in scikit-learn, as well as a Neural Network I coded from scratch, that classifies text as either **Bible** or **Not Bible**. I completed this project in August 2025, implementing a full pipeline for text preprocessing, feature extraction, model training, evaluation, and prediction.

## Project Overview

The goal is to predict whether a given piece of text comes from the Bible or not. The model uses TF-IDF vectorization to convert text into numerical features and a Random Forest Classifier, a Logistic Regression Classifier, and a Neural Network for prediction. The trained model can classify both dataset test samples and new custom inputs.

## Dataset

- Contains labeled samples of Bible verses and non-Bible text
- Split:
  - ~49,600 samples for training
  - ~12,400 samples for testing
- Classes: `Bible`, `Not Bible`

## Tools and Libraries

- Python  
- scikit-learn (`RandomForestClassifier`, `LogesticRegression`, `TfidfVectorizer`, `StandardScaler`)  
- pandas & NumPy (data handling)  
- joblib (model persistence)  

## Process and Methodology

### 1. Data Preprocessing
- Collected Bible and non-Bible text samples
  - NIV Bible: https://github.com/jadenzaleski/BibleTranslations/blob/master/NIV/NIV_bible.json
  - Non-Bible: pulled sentences from various books from Project Gutenberg (a mix of classics)
- Cleaned text (lowercasing, removing punctuation, tokenization)
- Converted text to numerical features using `TfidfVectorizer`

### 2. Feature Scaling
- Applied `StandardScaler` to normalize feature distributions  
- Saved the scaler as `scaler.joblib` for use during inference  

### 3. Model Training
- Trained a `RandomForestClassifier` with:
  - `n_estimators = 50`
  - `max_depth = None`
  - `n_jobs = -1` (parallelized across all CPU cores)
- Trained a `LogisticRegression` classifier with:
  - `max_iter` = 10000
- Trained a Neural Network that I built from scratch using base Python and NumPy with structure:
  - Layer_Dense with 80 neurons, l2 regularization = 1e-2
  - ReLU Activation
  - Layer_Dropout, dropout rate = 0.35 (prevent overfitting)
  - Layer_Dense with 40 neurons, l2 regularization = 1e-2
  - ReLU Activation
  - Layer_Dropout, dropout rate = 0.35
  - Layer_Dense with 2 neurons (bible or not)
  - Softmax probabilities activation + Loss_CategoricalCrossEntropy
  - Adam's Optimizer with learning rate = 0.00005, decay = 1e-3

### 4. Evaluation
- Evaluated accuracy on the test set
- Checked precision/recall and confusion matrix
- Tested predictions on custom inputs  

## Final Model Performance

- Test Set Accuracy:
  - Logistic Regression: ~90%
  - Random Forest: ~88%
  - Neural Network: ~93%
  - Ensemble Prediction (mode prediction of the 3 classifiers): ~93%
- Model demonstrates strong generalization to unseen text  

## Files in This Project

- data_scripts
  - bible_dataset_script.py
  - non_bible_dataset_script.py
  - merge.py
- models
  - NeuralNetwork.py
  - scaler.joblib
  - train_logreg.py
  - train_neuralnetwork.py
  - train_randomforest.py
  - vectorizer.joblib
- test_models.py
- README.md

## Timeline

8/18/25 - 8/31/25.  

## Future Improvements

- Try deep learning methods (e.g., LSTMs, Transformers like BERT)  
- Expand dataset with more diverse non-Bible text for robustness  
- Perform extensive hyperparameter optimization  
- Add cross-validation and ensemble methods for higher accuracy  
- Deploy model as a web app for real-time classification  

---
