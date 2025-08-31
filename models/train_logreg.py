from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

clf = LogisticRegression(max_iter = 10000)
clf.fit(X_train, y_train)

joblib.dump(clf, "logreg_bible_classifier.joblib")
print("Saved log reg classifier")