# TASK 1

import numpy as np

def normalize_data(data):
    # Convert data to numpy array for easier manipulation
    data_array = np.array(data)
    # Calculate mean and standard deviation for each feature
    mean = np.mean(data_array, axis=0)
    std_dev = np.std(data_array, axis=0)
    # Normalize each feature using Z-score normalization
    normalized_data = (data_array - mean) / std_dev
    return normalized_data.tolist()

def perceptron_train(data, learning_rate=0.1, max_epochs=1000):
    # Extract features and labels
    X = np.array([sample[:-1] for sample in data])
    y = np.array([sample[-1] for sample in data])
    num_features = X.shape[1]
    # Initialize weights (including bias term)
    weights = np.random.uniform(-1, 1, size=num_features+1)  # Add 1 for the bias term

    # Perceptron learning algorithm
    for epoch in range(max_epochs):
        for i in range(len(X)):
            # Add bias term to the current feature vector
            X_with_bias = np.concatenate(([1], X[i]))  # Add 1 as the first element for the bias term
            # Predict the label using current weights
            prediction = np.dot(X_with_bias, weights)
            # Update weights based on prediction error
            if prediction >= 0:
                prediction = 1
            else:
                prediction = 0
            error = y[i] - prediction
            weights += learning_rate * error * X_with_bias

    return weights.tolist()

def load_data(filename):
  with open (filename,'r') as file:
    samples = int(file.readline().strip())
    data = [list(map(float, line.strip().split())) for line in file.readlines()]
  return np.array(data)

train_data = load_data('train.txt')
normalized_data = normalize_data(train_data)
trained_weights = perceptron_train(normalized_data)
print("The weights on trained data are - ",trained_weights)

