# Script for Project 3 part 2
# ECE 4370: Engineering for Surgery
# Fall 2026
# Author:   Ryan Michael Nuera; ryan.michael.a.nuera@vanderbilt.edu
#           Julia Ke; julia.ke@vanderbilt.edu
#           Sabra Winston; sabra.m.winston@vanderbilt.edu
#
# Part 2: Multiclass probabilistic classifier using a multidimensional
# Gaussian fit to each class of the human activity training data.

import json
import numpy as np


class pca:
    """Principal component analysis of an NxM data matrix D (N samples, M features)."""

    def __init__(self, D):
        D = np.asarray(D, dtype=np.float64)
        self.mean = np.mean(D, axis=0)
        # Unbiased sample covariance (divides by N-1)
        cov = np.cov(D, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        # Sort modes from largest to smallest variance
        order = np.argsort(evals)[::-1]
        self.eigenvalues = evals[order]
        self.eigenvectors = evecs[:, order]  # columns are the principal modes

    def project(self, x):
        """Return PCA feature coordinates (NxM) of the NxM data matrix x."""
        return (np.asarray(x, dtype=np.float64) - self.mean) @ self.eigenvectors

    def reconstruct(self, f):
        """Map PCA feature coordinates back to the original data space."""
        return f @ self.eigenvectors.T + self.mean

    def squaredMahalanobis(self, f):
        """Squared Mahalanobis distance of each row of the NxM PCA feature matrix f.

        In the PCA basis the covariance is diagonal with the eigenvalues on the
        diagonal, so the distance reduces to sum_i f_i^2 / lambda_i.
        """
        f = np.atleast_2d(np.asarray(f, dtype=np.float64))
        return np.sum(f ** 2 / self.eigenvalues, axis=1)

    def logLikelihood(self, f):
        """Log of the multivariate Gaussian density evaluated at PCA coordinates f."""
        M = len(self.eigenvalues)
        return -0.5 * (self.squaredMahalanobis(f)
                       + np.sum(np.log(self.eigenvalues))
                       + M * np.log(2 * np.pi))


class gaussianClassifier:
    """Multiclass Bayes classifier with one multidimensional Gaussian per class."""

    def fit(self, D, y):
        self.classes = np.unique(y)
        self.models = [pca(D[y == c]) for c in self.classes]
        self.logPriors = np.log([np.mean(y == c) for c in self.classes])
        return self

    def logPosteriors(self, D):
        # Unnormalized log posterior: log p(x|c) + log P(c)
        return np.column_stack([m.logLikelihood(m.project(D)) + lp
                                for m, lp in zip(self.models, self.logPriors)])

    def predict(self, D):
        return self.classes[np.argmax(self.logPosteriors(D), axis=1)]


if __name__ == '__main__':
    # ---- Verify squaredMahalanobis with the provided test data ----
    f = open('SMDtest.json', 'rt')
    d = json.load(f)
    f.close()
    p = pca(np.array(d))
    f = p.project(np.zeros((3, 2), dtype=np.float64))
    print('SMD test (expect [0.27550597 0.27550597 0.27550597]):')
    print(p.squaredMahalanobis(f))

    # ---- Load human activity dataset ----
    f = open('Project3.json', 'rt')
    data = json.load(f)
    f.close()
    train_D = np.array(data['train_D'])
    train_y = np.array(data['train_y'])
    test_D = np.array(data['test_D'])
    test_y = np.array(data['test_y'])

    # ---- Train and evaluate the Gaussian classifier ----
    model = gaussianClassifier().fit(train_D, train_y)

    train_acc = np.mean(model.predict(train_D) == train_y)
    pred = model.predict(test_D)
    test_acc = np.mean(pred == test_y)

    print(f'\nTraining accuracy: {train_acc * 100:.2f}%')
    print(f'Testing accuracy:  {test_acc * 100:.2f}%')

    # Confusion matrix (rows = true class, columns = predicted class)
    classes = model.classes
    cm = np.array([[np.sum((test_y == t) & (pred == q)) for q in classes] for t in classes])
    print('\nTest confusion matrix (rows = true, cols = predicted), classes', classes)
    print(cm)
