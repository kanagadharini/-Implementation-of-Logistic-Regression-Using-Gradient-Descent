# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class LogisticRegressionGD:
    def _init_(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Gradient computation
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        y_probs = self.predict_proba(X)
        return np.where(y_probs >= 0.5, 1, 0)

# Example usage
if _name_ == "_main_":
    # Sample dataset: binary classification
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])  # Labels

    # Train model
    model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # Predict and evaluate
    predictions = model.predict(X)
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("Predictions:", predictions)
    print("Accuracy:", accuracy_score(y, predictions))
    print("\nClassification Report:\n", classification_report(y, predictions))
```

## Output:
![alt text](image-1.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

