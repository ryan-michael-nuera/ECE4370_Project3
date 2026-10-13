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

X_val = np.array(data["test_D"])
y_val = np.array(data["test_y"])

gammas = [1 / i for i in range(10, 81, 10)]

best_gamma = None
best_accuracy = -1

for gamma in gammas:
    svm = SVC(kernel="rbf", gamma=gamma)
    svm.fit(X_train, y_train)

    predictions = svm.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    print(f"Gamma = {gamma:.5f}, Accuracy = {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_gamma = gamma

final_svm = SVC(kernel="rbf", gamma=best_gamma)
final_svm.fit(X_train, y_train)

test_predictions = final_svm.predict(X_val)
test_accuracy = accuracy_score(y_val, test_predictions)

print("Best gamma:", best_gamma)
print("Testing accuracy:", test_accuracy)
