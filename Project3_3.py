# Script for Project 3 part 3
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import json
import numpy as np

with open("Project3.json", "r") as f:
    data = json.load(f)

X_train = np.array(data["train_D"])
y_train = np.array(data["train_y"])

# Validation set: used ONLY to choose gamma
X_val = np.array(data["validation_D"])
y_val = np.array(data["validation_y"])

# Testing set: used ONLY once, for the final accuracy
X_test = np.array(data["test_D"])
y_test = np.array(data["test_y"])

gammas = [1 / i for i in range(10, 81, 10)]

best_gamma = None
best_accuracy = -1
best_svm = None

for gamma in gammas:
    svm = SVC(kernel="rbf", gamma=gamma)
    svm.fit(X_train, y_train)

    predictions = svm.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    print(f"Gamma = 1/{round(1 / gamma)} ({gamma:.5f}), Validation accuracy = {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_gamma = gamma
        best_svm = svm

# Apply the model with the best gamma to the testing data
test_predictions = best_svm.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"\nBest gamma: 1/{round(1 / best_gamma)} ({best_gamma:.5f})")
print(f"Validation accuracy at best gamma: {best_accuracy * 100:.2f}%")
print(f"Testing accuracy: {test_accuracy * 100:.2f}%")
