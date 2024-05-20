# TASK 2

import numpy as np

def load_data(filename):
  with open (filename,'r') as file:
    samples = int(file.readline().strip())
    data = [list(map(float, line.strip().split())) for line in file.readlines()]
  return np.array(data)

#Function that takes data and weights as input
# def predict(X,weights):
#   X_with_bias = np.column_stack((np.ones(len(X)), X))
#   # print(shape[X_with_bias])
#   # print(shaweights))
#   predictions = np.dot(X_with_bias, weights)
#   return (predictions >= 0).astype(int)

def predict(data, weights):
    labels = []
    for sample in data:
        sample_with_bias = [1] + sample  # Adding a bias term to the sample
        prediction = sum([feature * weight for feature, weight in zip(sample_with_bias, weights)])
        label = 1 if prediction > 0 else 0  # Assuming binary classification
        labels.append(label)
    return labels

def evaluate_accuracy(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy

weights = load_data('weights.txt')
# weights = [-56428.17013026222, 29276.59308566843, 43309.77267552037, -49998.44036997994, -51312.5283953114]
test_data = load_data('test.txt')
y_pred = predict(test_data,weights)
print('Predicted Labels = ',y_pred)

