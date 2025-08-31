import numpy as np
import joblib

from NeuralNetwork import NeuralNetwork, Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossEntropy, Optimizer_Adam, Layer_Dropout

# ---------------------------
# LOAD DATA
# ---------------------------
print("Loading data...")

# Load sparse matrices
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

# Load labels
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Convert labels to one-hot
num_classes = len(np.unique(y_train))
y_train_oh = np.eye(num_classes)[y_train]
y_test_oh = np.eye(num_classes)[y_test]

print(f"Training samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {num_classes}")
print(f"Class dist: {np.bincount(y_train)}")

""" 
# NORMALIZATION BY HAND

X_train_mean = X_train.mean(axis = 0) # array with average input value over all samples
X_train_std = X_train.std(axis = 0)
# avoid division by 0
X_train_std[X_train_std == 0] = 1

X_train_normalized = (X_train - X_train_mean) / X_train_std
X_test_normalized = (X_test - X_train_mean) / X_train_std # still use training statistics
"""

# ---------------------------
# BUILD NETWORK
# ---------------------------
nn = NeuralNetwork([
    Layer_Dense(X_train.shape[1], 80, l2_lambda=1e-2),  # input -> hidden
    Activation_ReLU(),
    Layer_Dropout(0.35),
    Layer_Dense(80, 40, l2_lambda=1e-2),                # hidden -> hidden
    Activation_ReLU(),
    Layer_Dropout(0.35),
    Layer_Dense(40, num_classes),                        # hidden -> output
    Activation_Softmax_Loss_CategoricalCrossEntropy()    # softmax + crossentropy
])

# ---------------------------
# TRAIN NETWORK
# ---------------------------
optimizer = Optimizer_Adam(learning_rate=0.00005, decay=1e-3)

print("Training neural network...")
nn.train(
    X_train, y_train_oh,
    epochs=12,
    batch_size=128,
    optimizer=optimizer,
    verbose=True,
    X_val=X_test,
    y_val=y_test_oh
)

# ---------------------------
# SAVE TRAINED MODEL
# ---------------------------
joblib.dump(nn, "nn_model.joblib")
print("Neural network saved to nn_model.joblib")