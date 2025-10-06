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
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:  
*/
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X = np.random.randn(100, 2)
Y = (X[:, 0] + X[:, 1] > 0).astype(int)   


X = np.c_[np.ones(X.shape[0]), X]  


theta = np.zeros(X.shape[1])  
learning_rate = 0.1
epochs = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for i in range(epochs):
    z = np.dot(X, theta)
    h = sigmoid(z)
    gradient = np.dot(X.T, (h - Y)) / Y.size
    theta -= learning_rate * gradient


print("Final Parameters (theta):", theta)


def predict(X_new):
    X_new = np.c_[np.ones(X_new.shape[0]), X_new]
    return (sigmoid(np.dot(X_new, theta)) >= 0.5).astype(int)


Y_pred = predict(X[:, 1:])
accuracy = np.mean(Y_pred == Y)
print("Accuracy:", accuracy)


plt.figure(figsize=(6, 5))
plt.scatter(X[:, 1], X[:, 2], c=Y, cmap='bwr', label='Actual')
# Decision boundary
x_values = [np.min(X[:, 1]), np.max(X[:, 1])]
y_values = -(theta[0] + np.dot(theta[1], x_values)) / theta[2]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Logistic Regression using Gradient Descent')
plt.show()
```

## Output:
![logistic regression using gradient descent](sam.png)
![alt text](<Screenshot 2025-10-06 210451.png>)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

