from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=None,
    random_state=42,
    n_jobs=-1 # use all available cores
)

clf.fit(X_train, y_train)
joblib.dump(clf, "rf_bible_classifier.joblib")
print("Saved random forest clf")