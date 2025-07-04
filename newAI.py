import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Load data
data_path = r"D:\JAIN\Python\Asd-Adult-Dataset.csv"
data = pd.read_csv(data_path, na_values='?')
data.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Store for possible inverse_transform

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

# Perceptron initialization
weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.01
epochs = 100

def activation(x):
    return 1 if x >= 0 else 0  # Step function

def predict(X):
    return np.array([activation(np.dot(x, weights) + bias) for x in X])

# Perceptron training (still SGD style for pedagogical clarity)
for epoch in range(epochs):
    for i in range(len(X_train)):
        linear_output = np.dot(X_train[i], weights) + bias
        y_pred = activation(linear_output)
        error = y_train[i] - y_pred
        weights += learning_rate * error * X_train[i]
        bias += learning_rate * error

    # Accuracy per epoch
    y_epoch_pred = predict(X_test)
    epoch_acc = np.mean(y_epoch_pred == y_test)
    print(f"Epoch {epoch + 1} - Test Accuracy: {epoch_acc:.4f}")

# Final evaluation
y_pred = predict(X_test)
final_acc = np.mean(y_pred == y_test)
print("\nFinal Test Accuracy:", final_acc)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(2), yticklabels=range(2))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
