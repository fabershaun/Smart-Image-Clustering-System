import time
import json
import numpy as np
from numpy.linalg import norm
import pickle
from utils import evaluate_clustering_result


# Calculate the cosine similarity between two vectors
def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Load image feature vectors from a pickle file
def load_features(features_file):
    print(f'starting clustering images in {features_file}')
    with open(features_file, 'rb') as f:
        features = pickle.load(f)
    return features


# Normalize feature vectors. We chose this normalization method to reduce runtime to less than 15 seconds
def normalize_features(features):
    image_ids = list(features.keys())                                                            # Extract image names
    feature_vectors = np.array(list(features.values()))                                          # Convert feature vectors to a NumPy array
    feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)   # Normalization (each vector is divided by its own norm)
    return {image_ids[i]: feature_vectors[i] for i in range(len(image_ids))}                     # Return a dictionary mapping each image ID to its normalized feature vector


# Assign images to clusters based on cosine similarity
def assign_to_clusters(features, min_cluster_size, iterations):
    clusters = {}                                                   # Dictionary to store clusters (cluster_id → list of image IDs)
    centroids = {}                                                  # Dictionary to store centroids (cluster_id → feature vector)
    for iteration in range(iterations):
        print(f'Iteration {iteration + 1}/{iterations}...')
        for image_id, feature in features.items():
            similarities = {cluster_id: cosine(feature, centroids[cluster_id]) for cluster_id in centroids}  # Calculate similarity of the image to all existing cluster centroids
            max_cluster, max_similarity = max(similarities.items(), key=lambda x: x[1], default=(None, 0))   # Find the cluster with the highest similarity
            if max_similarity > 0.85:                     # If similarity is high enough, assign to existing cluster
                if max_cluster not in clusters:
                    clusters[max_cluster] = []            # Create an empty list for the new cluster
                clusters[max_cluster].append(image_id)
                centroids[max_cluster] = np.mean([features[i] for i in clusters[max_cluster]], axis=0)       # Update the cluster centroid by recalculating the mean of its members
            else:                                         # Create a new cluster with this image
                new_cluster_id = len(clusters)
                clusters[new_cluster_id] = [image_id]
                centroids[new_cluster_id] = feature       # Set the image feature as the initial centroid
    return clusters                                       # Return the final clusters


# Remove clusters that have fewer members than min_cluster_size
def filter_small_clusters(clusters, min_cluster_size):
    return {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}    # Keep only clusters where the number of images are equal or bigger than min_cluster_size


# Perform the full clustering process: loading, normalizing, clustering, and filtering
def cluster_data(features_file, min_cluster_size, iterations):
    features = load_features(features_file)                                     # Load image feature vectors from file
    features = normalize_features(features)                                     # Normalize feature vectors (L2 normalization)
    clusters = assign_to_clusters(features, min_cluster_size, iterations)       # Perform clustering
    clusters = filter_small_clusters(clusters, min_cluster_size)                # Remove small clusters
    return {cluster_id: members for cluster_id, members in clusters.items()}    # Return clusters in the required format {cluster_id: [list of image filenames]}






if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    result = cluster_data(config['features_file'],
                          config['min_cluster_size'],
                          config['max_iterations'])

    evaluation_scores = evaluate_clustering_result(config['labels_file'], result)  # implemented

    print(f'total time: {round(time.time() - start, 0)} sec')
