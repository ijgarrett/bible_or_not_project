# Bible-or-Not Text Classification

This project classifies text as **Bible** or **Not Bible** using multiple machine learning and deep learning approaches. It started with classical ML models in scikit-learn and a neural network coded from scratch in NumPy, and has since expanded to include a PyTorch-based Neural Network and a RoBERTa Transformer classifier.  

## Project Overview

The goal is to predict whether a given piece of text comes from the Bible or not.  
Models implemented:  
- **Logistic Regression** (scikit-learn)  
- **Random Forest Classifier** (scikit-learn)  
- **Neural Network from scratch** (NumPy implementation)  
- **Neural Network (PyTorch)**  
- **RoBERTa Transformer** (Hugging Face)  
All models use TF-IDF features or embeddings, depending on architecture, and can classify both dataset test samples and new custom inputs.  

## Dataset  

- **Bible text:** NIV Bible ([source](https://github.com/jadenzaleski/BibleTranslations/blob/master/NIV/NIV_bible.json))  
- **Non-Bible text:** Sentences sampled from various books from Project Gutenberg (classics).  
- **Split:**  
  - ~49,600 samples for training  
  - ~12,400 samples for testing  
- **Classes:** `Bible`, `Not Bible`  

## Tools and Libraries  
- **Python**  
- **scikit-learn**: `RandomForestClassifier`, `LogisticRegression`, `TfidfVectorizer`, `StandardScaler`  
- **NumPy & pandas**: data handling  
- **joblib**: model persistence  
- **PyTorch**: Neural Network implementation  
- **transformers (Hugging Face)**: RoBERTa fine-tuning  

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
#### Classical ML  
- **Random Forest**:
  - Implemented in train_randomforest.py
  - `n_estimators = 50`, `max_depth = None`, `n_jobs = -1`  
- **Logistic Regression**:
  - Implemented in train_logreg.py 
  - `max_iter = 10000`  

#### Neural Network (from scratch with NumPy)
- Implemented in train_pytorch_nn.py
- Dense layer with 80 neurons, L2 regularization = `1e-2`  
- ReLU + Dropout (0.35)  
- Dense layer with 40 neurons, L2 regularization = `1e-2`  
- ReLU + Dropout (0.35)  
- Output layer: 2 neurons, Softmax  
- Loss: Categorical Cross-Entropy  
- Optimizer: Adam (`lr=5e-5`, decay=`1e-3`)  

#### Neural Network (PyTorch)  
- Implemented in `pytorch_nn_model.py`
- Dense layer with 80 neurons
- ReLU + Dropout (0.35)  
- Dense layer with 40 neurons 
- ReLU + Dropout (0.35)  
- Output layer: 2 neurons, Softmax  
- Loss: Cross-Entropy Loss  
- Optimizer: Adam (`lr=4e-5`, weight_decay=`1e-3`)  

#### RoBERTa Transformer  
- Notebook: `train_roberta_bible.ipynb`  
- Uses Hugging Face `transformers` for fine-tuning  
- Optimized for Google Colab T4 GPU with the following hyperparameters:  
  - Epochs: 3  
  - Train batch size: 8  
  - Eval batch size: 16  
  - Warmup steps: 500  
  - Weight decay: 0.01  
  - Evaluation & checkpoint saving: each epoch  
  - Best model selection: based on accuracy  
  - Mixed precision (fp16): enabled  
  
### 4. Evaluation  

- Metrics: Accuracy, Precision, Recall, Confusion Matrix  
- Tested both on the dataset test split and custom user inputs  

## Final Model Performance

- Test Set Accuracy:
  - Logistic Regression: ~90%
  - Random Forest: ~88%
  - Neural Network from Scratch: ~93%
  - PyTorch Neural Network: ~93%
  - RoBERTa Transformer: ~98%
  - Ensemble Prediction (excluding RoBERTa): ~93%

## Files in This Project

- data_scripts
  - bible_dataset_script.py
  - non_bible_dataset_script.py
  - merge.py
  - prepare_data.py
- models
  - NeuralNetwork.py
  - scaler.joblib
  - train_logreg.py
  - train_neuralnetwork.py
  - train_pytorch_nn.py
  - train_randomforest.py
  - train_roberta.ipynb
  - vectorizer.joblib
- test_models.py
- README.md

## Timeline

8/18/25 - 9/27/25.  

## Future Improvements
- Deploy model as a web app for real-time classification  
---
