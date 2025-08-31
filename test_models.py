import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import mode

logreg = joblib.load("logreg_bible_classifier.joblib")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

y_pred_logreg = logreg.predict(X_test)

print("Log Reg Accuracy:", accuracy_score(y_pred_logreg, y_test))

nn = joblib.load("nn_model.joblib")

output_test = nn.forward(X_test)
y_pred_nn = np.argmax(output_test, axis = 1)
test_acc = np.mean(y_pred_nn == y_test)
print("Neural Network Accuracy:", accuracy_score(y_pred_nn, y_test))

rf = joblib.load("rf_bible_classifier.joblib")

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_pred_rf, y_test))

preds = np.stack([y_pred_rf, y_pred_logreg, y_pred_nn], axis = 1)
y_pred, _ = mode(preds, axis = 1) # mode returns mode and counts, the latter we don't need
y_pred = y_pred.flatten()

print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# USER INTERACTION
# ---------------------------
vectorizer = joblib.load("vectorizer.joblib")
scaler = joblib.load("scaler.joblib")

def classify_text(text):
    X_input = scaler.transform(vectorizer.transform([text]).toarray())
    
    pred_logreg = logreg.predict(X_input)[0] # take the value with [0]
    pred_rf = rf.predict(X_input)[0]
    output_nn = nn.forward(X_input)
    pred_nn = np.argmax(output_nn, axis=1)[0]

    preds = np.array([pred_logreg, pred_rf, pred_nn])
    ensemble_pred, _ = mode(preds, axis=None) # 1D array so no need for axis
    ensemble_pred = ensemble_pred.item() # turn into a scaler

    return pred_logreg, pred_rf, pred_nn, ensemble_pred

print("Type sentence to classify as bible or not (type \"stop\" to exit): ")
while True:
    user_input = input(">>> ")
    if user_input.lower() == "stop":
        print("Exiting")
        break
    
    pred_logreg, pred_rf, pred_nn, ensemble_pred = classify_text(user_input)

    print(f"LogReg: {'Bible' if pred_logreg == 1 else 'Not Bible'}")
    print(f"Random Forest: {'Bible' if pred_rf == 1 else 'Not Bible'}")
    print(f"Neural Net: {'Bible' if pred_nn == 1 else 'Not Bible'}")
    print(f"Ensemble: {'Bible' if ensemble_pred == 1 else 'Not Bible'}")