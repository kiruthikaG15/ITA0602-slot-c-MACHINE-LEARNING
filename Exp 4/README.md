
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)
print("Flower Classes:", iris.target_names)

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Normalize Dataset
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot Encoding
Y_train = np.eye(3)[y_train]

# Step 4: Initialize Weights and Biases
np.random.seed(42)

W1 = np.random.randn(4, 8) * 0.1
b1 = np.zeros((1, 8))

W2 = np.random.randn(8, 3) * 0.1
b2 = np.zeros((1, 3))

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# Step 5: Train ANN using Backpropagation
learning_rate = 0.1
epochs = 5000
m = len(X_train)

for epoch in range(epochs):

    # Forward Propagation
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    # Calculate Loss
    loss = -np.mean(
        np.sum(Y_train * np.log(A2 + 1e-9), axis=1)
    )

    # Backward Propagation
    dZ2 = (A2 - Y_train) / m

    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1)

    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update Weights and Biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 1000 == 0:
        print("Epoch:", epoch, "Loss:", round(loss, 4))

# Step 6: Test ANN
hidden = sigmoid(np.dot(X_test, W1) + b1)
output = softmax(np.dot(hidden, W2) + b2)

predictions = np.argmax(output, axis=1)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Step 7: Predict New Sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

new_sample_scaled = scaler.transform(new_sample)

hidden = sigmoid(np.dot(new_sample_scaled, W1) + b1)
output = softmax(np.dot(hidden, W2) + b2)

result = np.argmax(output, axis=1)[0]

print("\nNew Sample:", new_sample[0])
print("Predicted Flower:", iris.target_names[result])
