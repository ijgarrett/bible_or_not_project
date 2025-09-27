
import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from train_pytorch_nn import BibleNet
from transformers import pipeline

# ---------------------------
# LOAD MODELS AND DATA
# ---------------------------

logreg = joblib.load("logreg_bible_classifier.joblib")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
num_classes = len(np.unique(y_test))

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorch_nn = BibleNet(X_test.shape[1], num_classes).to(device)
pytorch_nn.load_state_dict(torch.load("pytorch_nn_model.pth", map_location=device))
pytorch_nn.eval()  # Set to evaluation mode

# Load other models
nn = joblib.load("nn_model.joblib")
rf = joblib.load("rf_bible_classifier.joblib")

# Load RoBERTa model
print("Loading RoBERTa model...")
roberta_classifier = pipeline(
    "text-classification",
    model="roberta_bible_model",  # Path to your downloaded model folder
    tokenizer="roberta_bible_model",
    device=0 if torch.cuda.is_available() else -1,
    framework="pt"  # Force PyTorch framework to avoid TensorFlow issues
)
print("RoBERTa model loaded!")

# ---------------------------
# EVALUATE ALL MODELS
# ---------------------------
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_pred_logreg, y_test))

# Original neural network
output_test = nn.forward(X_test)
y_pred_nn = np.argmax(output_test, axis=1)
print("Neural Network (from scratch) Accuracy:", accuracy_score(y_pred_nn, y_test))

# PyTorch neural network
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    pytorch_output = pytorch_nn(X_test_tensor)
    y_pred_pytorch = torch.argmax(pytorch_output, dim=1).cpu().numpy()
print("PyTorch Neural Network Accuracy:", accuracy_score(y_pred_pytorch, y_test))

# Random Forest
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_pred_rf, y_test))

# RoBERTa model - do in google colab

# Ensemble with all 4 models 
preds_all = np.stack([y_pred_rf, y_pred_logreg, y_pred_nn, y_pred_pytorch], axis=1)
y_pred_ensemble_all, _ = mode(preds_all, axis=1)
y_pred_ensemble_all = y_pred_ensemble_all.flatten()
print("Ensemble Accuracy (5 models including RoBERTa):", accuracy_score(y_test, y_pred_ensemble_all))

# ---------------------------
# USER INTERACTION
# ---------------------------
vectorizer = joblib.load("vectorizer.joblib")
scaler = joblib.load("scaler.joblib")

def classify_text(text):
    X_input = scaler.transform(vectorizer.transform([text]).toarray())
    
    # Get predictions from traditional models
    pred_logreg = logreg.predict(X_input)[0]
    pred_rf = rf.predict(X_input)[0]
    
    # Original neural network
    output_nn = nn.forward(X_input)
    pred_nn = np.argmax(output_nn, axis=1)[0]
    
    # PyTorch neural network
    X_input_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        pytorch_output = pytorch_nn(X_input_tensor)
        pred_pytorch = torch.argmax(pytorch_output, dim=1).cpu().numpy()[0]
    
    # RoBERTa prediction (works directly with raw text)
    roberta_result = roberta_classifier(text)
    # Map labels: LABEL_0 = Not Bible (0), LABEL_1 = Bible (1)
    pred_roberta = 1 if roberta_result[0]['label'] == 'LABEL_1' else 0
    roberta_confidence = roberta_result[0]['score']
    
    # Ensemble prediction with all 5 models
    preds = np.array([pred_logreg, pred_rf, pred_nn, pred_pytorch, pred_roberta])
    ensemble_pred, _ = mode(preds, axis=None)
    ensemble_pred = ensemble_pred.item()

    return pred_logreg, pred_rf, pred_nn, pred_pytorch, pred_roberta, roberta_confidence, ensemble_pred

print("\nType sentence to classify as bible or not (type \"stop\" to exit): ")
while True:
    user_input = input(">>> ")
    if user_input.lower() == "stop":
        print("Exiting")
        break
    
    pred_logreg, pred_rf, pred_nn, pred_pytorch, pred_roberta, roberta_conf, ensemble_pred = classify_text(user_input)

    print(f"Logistic Regression: {'Bible' if pred_logreg == 1 else 'Not Bible'}")
    print(f"Random Forest: {'Bible' if pred_rf == 1 else 'Not Bible'}")
    print(f"Neural Net (Sklearn): {'Bible' if pred_nn == 1 else 'Not Bible'}")
    print(f"PyTorch Neural Net: {'Bible' if pred_pytorch == 1 else 'Not Bible'}")
    print(f"RoBERTa: {'Bible' if pred_roberta == 1 else 'Not Bible'} (confidence: {roberta_conf:.3f})")
    print(f"Ensemble (5 models): {'Bible' if ensemble_pred == 1 else 'Not Bible'}")
    print("-" * 50)