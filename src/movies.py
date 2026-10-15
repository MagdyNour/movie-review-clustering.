import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Load dataset from the project's local data folder
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "data", "movie_reviews.csv")
reviews_df = pd.read_csv(data_path)

# ========== PROBLEM 2a: BAG-OF-WORDS ==========
#these prints just make the output easier to read (suggested by chatGPT)
print("PROBLEM 2a: BAG-OF-WORDS IMPLEMENTATION")
print("=" * 50)

# Data preprocessing
reviews_lowercase = reviews_df['Review'].str.lower()
reviews_no_punct = reviews_lowercase.str.replace(r'[^\w\s]', '', regex=True)
stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
reviews_clean = reviews_no_punct.apply(lambda text: ' '.join([word for word in text.split() if word not in stop_words]))

reviews_clean = reviews_clean.drop_duplicates()
reviews_df = reviews_df.loc[reviews_clean.index]

# Build vocabulary
all_words = []
for review in reviews_clean:
    all_words.extend(review.split())
vocabulary = sorted(set(all_words))

# Create BoW vectors
bow_vectors = []
for review in reviews_clean:
    words = review.split()
    vector = [words.count(word) for word in vocabulary]
    bow_vectors.append(vector)

bow_vectors_array = np.array(bow_vectors)

print(f"Vocabulary size: {len(vocabulary)}")
print(f"Number of reviews: {len(bow_vectors)}")
print(f"BoW vectors shape: {bow_vectors_array.shape}")

# ========== PROBLEM 2b: HAC IMPLEMENTATION ==========
print("\n" + "="*70)
print("PROBLEM 2b: HAC IMPLEMENTATION")
print("="*70)

def compute_distance(data, cluster1, cluster2):
    total_distance = 0
    count = 0
    for i in cluster1:
        for j in cluster2:
            distance = np.sqrt(sum((data[i][d] - data[j][d]) ** 2 for d in range(len(data[0]))))
            total_distance += distance
            count += 1
    return total_distance / count if count > 0 else float('inf')

def simple_hac(data, k):
    clusters = [[i] for i in range(len(data))]
    while len(clusters) > k:
        min_distance = float('inf')
        closest_pair = (0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = compute_distance(data, clusters[i], clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j)
        clusters[closest_pair[0]].extend(clusters[closest_pair[1]])
        del clusters[closest_pair[1]]
    return clusters

# ========== PROBLEM 2c: CLUSTERING COMPARISON ==========
print("\n" + "="*70)
print("PROBLEM 2c: CLUSTERING COMPARISON")
print("="*70)

#kmeans without sklearn (use ai because i had too many errors in my loopings) 
def k_means(data, k, max_iters=100):
    n_samples = data.shape[0]
    random_indices = random.sample(range(n_samples), k)
    centroids = data[random_indices]
    
    for iteration in range(max_iters):
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            distances = np.sqrt(np.sum((data[i] - centroids) ** 2, axis=1))
            labels[i] = np.argmin(distances)
        
        new_centroids = np.zeros_like(centroids)
        for cluster_id in range(k):
            cluster_points = data[labels == cluster_id]
            if len(cluster_points) > 0:
                new_centroids[cluster_id] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[cluster_id] = data[random.randint(0, n_samples-1)]
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels


#i had no idea how to calculate the silhouette scores so i also used ai for the core of this function after reading on wikipedia about how its calculated ofcourse.
def silhouette_score_custom(data, labels):
    n = len(data)
    if len(np.unique(labels)) <= 1:
        return 0
    
    silhouette_scores = []
    for i in range(n):
        cluster_mask = (labels == labels[i])
        cluster_points = data[cluster_mask]
        if len(cluster_points) > 1:
            a_i = np.mean([np.sqrt(np.sum((data[i] - x)**2)) for x in cluster_points if not np.array_equal(x, data[i])])
        else:
            a_i = 0
        b_i = float('inf')
        for cluster_id in np.unique(labels):
            if cluster_id != labels[i]:
                other_cluster_points = data[labels == cluster_id]
                avg_dist = np.mean([np.sqrt(np.sum((data[i] - x)**2)) for x in other_cluster_points])
                if avg_dist < b_i:
                    b_i = avg_dist
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores)

def compute_cluster_metrics(data, labels, centroids=None):
    k = len(np.unique(labels))
    cardinalities = []
    magnitudes = []
    
    for cluster_id in range(k):
        cluster_points = data[labels == cluster_id]
        cardinality = len(cluster_points)
        cardinalities.append(cardinality)
        if centroids is not None and len(cluster_points) > 0:
            magnitude = np.sum([np.sum((point - centroids[cluster_id])**2) for point in cluster_points])
            magnitudes.append(magnitude)
        else:
            magnitudes.append(0)
    
    avg_cardinality = np.mean(cardinalities)
    avg_magnitude = np.mean(magnitudes)
    
    return avg_cardinality, avg_magnitude, cardinalities

# Run k-means clustering
k_values = [2, 3, 5]
kmeans_results = {}

print("K-MEANS CLUSTERING RESULTS:")
print("k | Avg Cardinality | Avg Magnitude | Silhouette Score")
print("-" * 55)

for k in k_values:
    centroids, kmeans_labels = k_means(bow_vectors_array, k)
    avg_cardinality, avg_magnitude, cardinalities = compute_cluster_metrics(bow_vectors_array, kmeans_labels, centroids)
    sil_score = silhouette_score_custom(bow_vectors_array, kmeans_labels)
    kmeans_results[k] = {
        'labels': kmeans_labels, 
        'cardinalities': cardinalities,
        'avg_cardinality': avg_cardinality,
        'avg_magnitude': avg_magnitude,
        'silhouette': sil_score
    }
    print(f"{k} | {avg_cardinality:14.1f} | {avg_magnitude:13.1f} | {sil_score:.4f}")

# Run HAC clustering (only k=2 to save time)
print("\nHAC CLUSTERING RESULTS (k=2 only):")
print("k | Avg Cardinality | Avg Magnitude | Silhouette Score")
print("-" * 55)

hac_results = {}
for k in [2]:  # Only test k=2 to save time
    hac_clusters = simple_hac(bow_vectors, k)
    hac_labels = np.zeros(len(bow_vectors))
    for cluster_id, cluster in enumerate(hac_clusters):
        for point_idx in cluster:
            hac_labels[point_idx] = cluster_id
    
    centroids = []
    for cluster_id in range(k):
        cluster_points = bow_vectors_array[hac_labels == cluster_id]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
        else:
            centroid = np.zeros(bow_vectors_array.shape[1])
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    avg_cardinality, avg_magnitude, cardinalities = compute_cluster_metrics(bow_vectors_array, hac_labels, centroids)
    sil_score = silhouette_score_custom(bow_vectors_array, hac_labels)
    hac_results[k] = {
        'labels': hac_labels,
        'cardinalities': cardinalities,
        'avg_cardinality': avg_cardinality,
        'avg_magnitude': avg_magnitude,
        'silhouette': sil_score
    }
    print(f"{k} | {avg_cardinality:14.1f} | {avg_magnitude:13.1f} | {sil_score:.4f}")

# ========== PROBLEM 2d: MDS VISUALIZATION ==========
print("\n" + "="*70)
print("PROBLEM 2d: MDS VISUALIZATION")
print("="*70)

def simple_mds(dissimilarities, n_components=2):
    n = dissimilarities.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dissimilarities ** 2) @ J
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    X = eigenvectors[:, :n_components] @ np.diag(np.sqrt(eigenvalues[:n_components]))
    return X

def bow_to_dissimilarity(bow_vectors):
    bow_array = np.array(bow_vectors)
    n = bow_array.shape[0]
    dissimilarities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dot_product = np.dot(bow_array[i], bow_array[j])
            norm_i = np.linalg.norm(bow_array[i])
            norm_j = np.linalg.norm(bow_array[j])
            if norm_i > 0 and norm_j > 0:
                cosine_sim = dot_product / (norm_i * norm_j)
                dissimilarities[i, j] = 1 - cosine_sim
            else:
                dissimilarities[i, j] = 1
    return dissimilarities

dissimilarity_matrix = bow_to_dissimilarity(bow_vectors)
mds_coords = simple_mds(dissimilarity_matrix, n_components=2)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    mds_coords[:, 0], mds_coords[:, 1],
    c=reviews_df['Label'].map({'Positive': 0, 'Negative': 1}),
    cmap='viridis', alpha=0.7
)
plt.colorbar(scatter, label='Label (0=Positive, 1=Negative)')
plt.title('MDS Visualization of Movie Reviews (Colored by True Sentiment)')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.grid(True, alpha=0.3)
plt.show()

# ========== PROBLEM 2e: IDF ANALYSIS ==========
print("\n" + "="*70)
print("PROBLEM 2e: INVERSE DOCUMENT FREQUENCY")
print("="*70)

def compute_idf(bow_vectors, vocabulary):
    bow_array = np.array(bow_vectors)
    document_frequency = np.sum(bow_array > 0, axis=0)
    total_documents = bow_array.shape[0]
    idf = np.log(total_documents / (document_frequency + 1e-10))  # Avoid division by zero
    idf_df = pd.DataFrame({'word': vocabulary, 'document_frequency': document_frequency, 'idf': idf})
    return idf_df

idf_results = compute_idf(bow_vectors, vocabulary)
idf_sorted = idf_results.sort_values('idf', ascending=False)

print("TOP 10 HIGHEST IDF WORDS (most discriminative):")
print(idf_sorted.head(10)[['word', 'idf', 'document_frequency']])
print("\nTOP 10 LOWEST IDF WORDS (most common):")
print(idf_sorted.tail(10)[['word', 'idf', 'document_frequency']])

# ========== PROBLEM 2f: CLASSIFICATION WITH FORWARD SELECTION ==========
print("\n" + "="*70)
print("PROBLEM 2f: NAIVE BAYES CLASSIFICATION WITH FORWARD SELECTION")
print("="*70)

def simple_naive_bayes(X_train, y_train, X_test):
    """CORRECTED Naive Bayes implementation"""
    n_samples, n_features = X_train.shape
    
    # Class probabilities
    class_1_count = np.sum(y_train == 1)
    class_0_count = len(y_train) - class_1_count
    p_class_1 = (class_1_count + 1) / (len(y_train) + 2)
    p_class_0 = (class_0_count + 1) / (len(y_train) + 2)
    
    # Feature probabilities - FIXED
    alpha = 1
    
    # For class 1
    features_class_1 = X_train[y_train == 1]
    total_words_class_1 = np.sum(features_class_1)
    p_features_given_1 = (np.sum(features_class_1, axis=0) + alpha) / (total_words_class_1 + alpha * n_features)
    
    # For class 0  
    features_class_0 = X_train[y_train == 0]
    total_words_class_0 = np.sum(features_class_0)
    p_features_given_0 = (np.sum(features_class_0, axis=0) + alpha) / (total_words_class_0 + alpha * n_features)
    
    # Predictions
    predictions = []
    for x in X_test:
        #in the folowing lines we ad the small value so that i dont get error with log of 0 
        log_p_1 = np.log(p_class_1) + np.sum(x * np.log(p_features_given_1 + 1e-10))
        log_p_0 = np.log(p_class_0) + np.sum(x * np.log(p_features_given_0 + 1e-10))
        predictions.append(1 if log_p_1 > log_p_0 else 0)
    
    return np.array(predictions)

def simple_forward_selection(X, y, vocabulary, max_features=10):
    """Simple forward feature selection"""
    selected_features = []
    remaining_features = list(range(X.shape[1]))
    selected_words = []
    accuracies = []
    
    print("Forward Selection Progress:")
    print("-" * 40)
    
    for step in range(min(max_features, len(remaining_features))):
        best_acc = -1
        best_feat = None
        
        for feature in remaining_features:
            current_feats = selected_features + [feature]
            X_current = X[:, current_feats]
            
            # Simple train/test split (80/20)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X_current[:split_idx], X_current[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train and evaluate
            y_pred = simple_naive_bayes(X_train, y_train, X_test)
            accuracy = np.mean(y_pred == y_test)
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_feat = feature
        
        # Add the best feature
        selected_features.append(best_feat)
        selected_words.append(vocabulary[best_feat])
        remaining_features.remove(best_feat)
        accuracies.append(best_acc)
        
        print(f"Step {step+1}: Selected '{vocabulary[best_feat]}' - Accuracy: {best_acc:.3f}")
    
    return selected_features, selected_words, accuracies

# Convert labels to binary
y_binary = reviews_df['Label'].map({'Positive': 1, 'Negative': 0}).values

# Run forward selection
selected_feats, selected_words, accuracies = simple_forward_selection(
    bow_vectors_array, y_binary, vocabulary, max_features=10
)

# Print final results
print("\nFINAL SELECTED FEATURES:")
print("-" * 30)
for i, (word, acc) in enumerate(zip(selected_words, accuracies)):
    print(f"{i+1:2d}. {word:15s} | Accuracy: {acc:.3f}")

# Analyze IDF of selected features
print("\nIDF ANALYSIS OF SELECTED FEATURES:")
print("-" * 35)
selected_idf_values = []
for i, feat_idx in enumerate(selected_feats):
    word = vocabulary[feat_idx]
    idf_val = idf_results[idf_results['word'] == word]['idf'].values[0]
    selected_idf_values.append(idf_val)
    print(f"{word:15s} | IDF: {idf_val:.3f}")

print(f"\nAverage IDF of selected features: {np.mean(selected_idf_values):.3f}")
print(f"Overall average IDF: {idf_results['idf'].mean():.3f}")

print("\n" + "="*70)
print("PROBLEM 2 COMPLETED SUCCESSFULLY!")
print("="*70)