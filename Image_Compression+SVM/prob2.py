# QUESTION 2

# Task 1 (a)

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

iris = datasets.load_iris(as_frame = True)
# print(iris)
# Extract petal length and petal width features
X = iris.data[['petal length (cm)', 'petal width (cm)']]
# Extract target classes
y = iris.target

# Filter data to include only 'setosa' and 'versicolor' classes
X_filtered = X[y != 2]
y_filtered = y[y != 2]
# print(X_filtered)

# Normalization
X_mean = np.mean(X_filtered, axis=0)
X_std = np.std(X_filtered, axis=0)
X_normalized = (X_filtered - X_mean) / X_std

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_filtered, test_size=0.2, random_state=55)

print("Shapes after filtering and splitting:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Task 1(b):

# Train LinearSVC on the training data
clf = LinearSVC()
clf.fit(X_train, y_train)

# Plot decision boundary on the training data
x0_min, x0_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
x1_min, x1_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1

xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.02),
                       np.arange(x1_min, x1_max, 0.02))
Z = clf.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)

# Plot decision boundary and training data
plt.contourf(xx0, xx1, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Decision boundary on train data')
plt.show()

# Generate scatterplot of the test data along with original decision boundary
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Test data with original decision boundary')

# Plot original decision boundary
plt.contour(xx0, xx1, Z, colors='k', linestyles=['-'], levels=[0])

plt.show()

# Task 2(a)

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate synthetic dataset with 500 data points and 5% noise
X, y = make_moons(n_samples=500, noise=0.05, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot the synthetic dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset with 5% Noise')
plt.show()

# Task 2(b):

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Create meshgrid for plotting decision boundaries
def plot_decision_boundary(ax, clf, X, y, title):
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)

# SVM with Linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# SVM with Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3)  # You can adjust the degree
svm_poly.fit(X_train, y_train)

# SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', gamma='auto')  # You can adjust the gamma
svm_rbf.fit(X_train, y_train)

# Plot decision boundaries for each kernel
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_decision_boundary(axs[0], svm_linear, X, y, 'Linear Kernel')
plot_decision_boundary(axs[1], svm_poly, X, y, 'Polynomial Kernel')
plot_decision_boundary(axs[2], svm_rbf, X, y, 'RBF Kernel')
plt.show()

# Task 2(c)(d):

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define the parameter grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 50, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10,50, 100]}

# Initialize the SVM model with RBF kernel
svm_rbf = SVC(kernel='rbf')

# Perform grid search
grid_search = GridSearchCV(svm_rbf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Predict labels for test data using the best model
y_pred = best_model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot decision boundary of the best model
plt.figure(figsize=(8, 6))
plot_decision_boundary(plt.gca(), best_model, X_train, y_train, 'Best RBF Kernel SVM')
plt.show()
