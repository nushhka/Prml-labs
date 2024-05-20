# QUESTION 1

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

image = Image.open('test.png')
# Convert image to numpy array
image_np = np.array(image)
# Reshape the image to a 2D array of pixels
image_reshaped = image_np.reshape(-1, 3)
print(image_reshaped.shape)
# Shuffle the pixels
image_reshaped_sample = shuffle(image_reshaped, random_state=0)[:image_reshaped.shape[0]]

# TASK (a)

def computeCentroid(features):
    num_features = len(features)
    dimension = len(features[0])
    # Initialize a list to accumulate the sum of each dimension
    centroid = [0] * dimension

    # Sum up each dimension separately
    for feature in features:
        for i in range(dimension):
            centroid[i] += feature[i]

    # Calculate the mean for each dimension
    for i in range(dimension):
        centroid[i] /= num_features

    # Return the centroid as a tuple
    return tuple(centroid)

# Compute the centroid of the image
centroid = computeCentroid(image_reshaped)

# Print the centroid
print("Centroid:", centroid)

# TASK (b)

def mykmeans(X, k):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)] #random centroids generated
    while True:
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)   # Assign each data point to the nearest centroid
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # Stopping condn
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids 

# TASK (c)

def compress_image(image, centroids, labels):
    compressed_image = centroids[labels]
    compressed_image = compressed_image.astype(np.uint8)
    compressed_image = compressed_image.reshape(image.shape)
    return compressed_image

# Define different values of k
k_values = [2, 4, 8]

# Create a figure to display images
plt.figure(figsize=(12, 4))

# Perform k-means for each value of k using mykmeans
for i, k in enumerate(k_values):
    centroids = mykmeans(image_reshaped, k)
    labels = np.argmin(np.linalg.norm(image_reshaped[:, np.newaxis] - centroids, axis=2), axis=1)
    compressed_image = compress_image(image_np, centroids, labels)
    # Add subplot for each compressed image
    plt.subplot(1, len(k_values), i + 1)
    plt.imshow(compressed_image)
    plt.title(f'k={k}')
    plt.axis('off')

plt.suptitle('Compressed Images with Different Values of k')
# Show the figure with all compressed images
plt.show()

# TASK (d)

# Create a figure to display images
plt.figure(figsize=(12, 4))

# Perform k-means for each value of k using scikit-learn implementation
for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(image_reshaped)
    centroids = kmeans.cluster_centers_
    compressed_image = centroids[labels].reshape(image_np.shape)
    # Add subplot for each compressed image
    plt.subplot(1, len(k_values), i + 1)
    plt.imshow(compressed_image.astype(np.uint8))
    plt.title(f'k={k}')
    plt.axis('off')

# Add a title to the figure
plt.suptitle('Compressed Images with Different Values of k (Using scikit-learn KMeans)')

# Show the figure with all compressed images
plt.show()

# Task (e):

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import io

def spatial_coherence_clustering(image, k, spatial_weight=0.5):
    # Flatten the image into a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Get the spatial coordinates of each pixel
    h, w, _ = image.shape
    spatial_coords = np.array([[i, j] for i in range(h) for j in range(w)])

    # Normalize spatial coordinates to [0, 1]
    spatial_coords_norm = spatial_coords / max(h, w)

    # Combine color and spatial coordinates
    combined_features = np.concatenate([pixels, spatial_weight * spatial_coords_norm], axis=1)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(combined_features)

    # Reconstruct the image based on cluster centroids
    reconstructed_image = np.zeros_like(pixels)
    for i in range(k):
        cluster_mean_color = kmeans.cluster_centers_[i][:3]
        cluster_pixels_indices = np.where(labels == i)
        reconstructed_image[cluster_pixels_indices] = cluster_mean_color

    # Reshape the reconstructed image to the original shape
    reconstructed_image = reconstructed_image.reshape(image.shape)

    return reconstructed_image

# Load the test image
image = io.imread('test.png')

# Define values of k and spatial_weight to test
k_values = [2, 4, 8]
spatial_weights = [0.2, 0.5, 0.8]  # Adjust as needed

# Plot original image
plt.figure(figsize=(15, 10))

# # Plot original image in the first row
# plt.subplot(2, len(k_values), 1)
# plt.imshow(image)
# plt.title('Original Image')
# plt.axis('off')

# Loop over each combination of k and spatial_weight
for i, k in enumerate(k_values):
    for j, spatial_weight in enumerate(spatial_weights):
        # Plot compressed images in subsequent rows
        index = i * len(spatial_weights) + j + 1
        plt.subplot(3, len(k_values), index)
        compressed_image = spatial_coherence_clustering(image, k, spatial_weight)
        plt.imshow(compressed_image)
        plt.title(f'k={k}, spatial_weight={spatial_weight}')
        plt.axis('off')

plt.tight_layout()
plt.show()

