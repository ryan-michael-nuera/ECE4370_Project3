# Script for Project 3 part 1
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu

import json
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATA
# ============================================================

with open("Project3.json", "r") as f:
    data = json.load(f)

# Check available keys first
print("JSON keys:", data.keys())

X_train = np.asarray(data["train_D"], dtype=np.float64)
y_train = np.asarray(data["train_y"])

X_val = np.asarray(data["validation_D"], dtype=np.float64)
y_val = np.asarray(data["validation_y"])

X_test = np.asarray(data["test_D"], dtype=np.float64)
y_test = np.asarray(data["test_y"])

print("\nDataset shapes:")
print("Training:  ", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:   ", X_test.shape, y_test.shape)


# ============================================================
# 2. PREPROCESS LABELS
# ============================================================

# Convert arbitrary class labels into 0, 1, ..., K-1
classes = np.unique(y_train)
num_classes = len(classes)

class_to_index = {
    label: i for i, label in enumerate(classes)
}

y_train_idx = np.array([class_to_index[y] for y in y_train])
y_val_idx = np.array([class_to_index[y] for y in y_val])
y_test_idx = np.array([class_to_index[y] for y in y_test])

print("\nClasses:", classes)
print("Number of classes:", num_classes)


# ============================================================
# 3. STANDARDIZE FEATURES
# ============================================================

# IMPORTANT:
# Compute mean/std ONLY from the training set.
# Do not use validation or test data to calculate these values.

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# Prevent division by zero
std[std == 0] = 1.0

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# ============================================================
# 4. SOFTMAX FUNCTION
# ============================================================

def softmax(z):
    """
    Convert linear scores into class probabilities.
    """

    # Numerical stability
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ============================================================
# 5. CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy(y_true, probabilities):
    """
    Multiclass cross-entropy loss.
    """

    n = len(y_true)

    # Probability assigned to the correct class
    correct_probs = probabilities[np.arange(n), y_true]

    # Prevent log(0)
    correct_probs = np.clip(correct_probs, 1e-12, 1.0)

    loss = -np.mean(np.log(correct_probs))

    return loss


# ============================================================
# 6. PREDICTION FUNCTION
# ============================================================

def predict(X, W, b):
    """
    Predict class index using the linear model.
    """

    # Linear model:
    # scores = XW + b
    scores = X @ W + b

    probabilities = softmax(scores)

    predictions = np.argmax(probabilities, axis=1)

    return predictions


# ============================================================
# 7. ACCURACY FUNCTION
# ============================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ============================================================
# 8. INITIALIZE THE LINEAR MODEL
# ============================================================

num_features = X_train.shape[1]

rng = np.random.default_rng(42)

# Weight matrix:
# number of features x number of classes
W = rng.normal(
    loc=0.0,
    scale=0.01,
    size=(num_features, num_classes)
)

# One bias for each class
b = np.zeros(num_classes)

print("\nWeight matrix shape:", W.shape)
print("Bias vector shape:", b.shape)


# ============================================================
# 9. TRAINING SETTINGS
# ============================================================

learning_rate = 0.05
epochs = 1000

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# ============================================================
# 10. TRAIN THE MULTICLASS LINEAR MODEL
# ============================================================

n_train = X_train.shape[0]

for epoch in range(epochs):

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    # Linear scores
    train_scores = X_train @ W + b

    # Convert scores into probabilities
    train_probs = softmax(train_scores)


    # --------------------------------------------------------
    # TRAINING LOSS
    # --------------------------------------------------------

    train_loss = cross_entropy(
        y_train_idx,
        train_probs
    )


    # --------------------------------------------------------
    # CREATE ONE-HOT LABEL MATRIX
    # --------------------------------------------------------

    y_onehot = np.zeros(
        (n_train, num_classes)
    )

    y_onehot[
        np.arange(n_train),
        y_train_idx
    ] = 1


    # --------------------------------------------------------
    # BACKPROPAGATION / GRADIENT CALCULATION
    # --------------------------------------------------------

    # Gradient of cross-entropy + softmax
    error = train_probs - y_onehot

    # Weight gradient
    dW = (X_train.T @ error) / n_train

    # Bias gradient
    db = np.mean(error, axis=0)


    # --------------------------------------------------------
    # GRADIENT DESCENT UPDATE
    # --------------------------------------------------------

    W = W - learning_rate * dW
    b = b - learning_rate * db


    # --------------------------------------------------------
    # TRAINING ACCURACY
    # --------------------------------------------------------

    train_pred = predict(
        X_train,
        W,
        b
    )

    train_acc = accuracy(
        y_train_idx,
        train_pred
    )


    # ========================================================
    # VALIDATION PROCESS
    # ========================================================

    # Do NOT update W or b using validation data.

    val_scores = X_val @ W + b

    val_probs = softmax(val_scores)

    val_loss = cross_entropy(
        y_val_idx,
        val_probs
    )

    val_pred = np.argmax(
        val_probs,
        axis=1
    )

    val_acc = accuracy(
        y_val_idx,
        val_pred
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)


    # --------------------------------------------------------
    # PRINT TRAINING PROGRESS
    # --------------------------------------------------------

    if epoch % 50 == 0 or epoch == epochs - 1:

        print(
            f"Epoch {epoch:4d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )


# ============================================================
# 11. FINAL TESTING PROCESS
# ============================================================

# The test set has not been used to train the model.

test_scores = X_test @ W + b

test_probs = softmax(test_scores)

test_pred = np.argmax(
    test_probs,
    axis=1
)

test_loss = cross_entropy(
    y_test_idx,
    test_probs
)

test_acc = accuracy(
    y_test_idx,
    test_pred
)


print("\n======================================")
print("FINAL MODEL PERFORMANCE")
print("======================================")

print(f"Training accuracy:   {train_accuracies[-1] * 100:.2f}%")
print(f"Validation accuracy: {val_accuracies[-1] * 100:.2f}%")
print(f"Testing accuracy:    {test_acc * 100:.2f}%")
print(f"Testing loss:        {test_loss:.4f}")


# ============================================================
# 12. PLOT TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 13. PLOT TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    np.array(train_accuracies) * 100,
    label="Training Accuracy"
)

plt.plot(
    np.array(val_accuracies) * 100,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test_idx,
    test_pred,
    display_labels=classes,
    cmap="Blues"
)

plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.show()