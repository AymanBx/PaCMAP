import numpy as np
from pacmap import PaCMAP
from DataLoader import DatasetLoader
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

def run_trustworthiness(db_name, X_train, y_train, X_test, y_test, embedding_func, n_neighbors_list=[5, 10, 15]):
    """
    Computes trustworthiness between original data and reduced embeddings.
    """
    
    print("Run Trustworthiness on ==> ", db_name)

    # Flatten if necessary
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print("X_train flattened shape:", X_train_flat.shape)
    print("X_test flattened shape:", X_test_flat.shape)

    # Generate embeddings
    print("Generating embeddings...")
    X_train_embedded, X_test_embedded = embedding_func(X_train_flat, X_test_flat)

    # Evaluate trustworthiness
    for k in n_neighbors_list:
        tw_train = trustworthiness(X_train_flat, X_train_embedded, n_neighbors=k)
        tw_test = trustworthiness(X_test_flat, X_test_embedded, n_neighbors=k)
        print(f"Trustworthiness (k={k}) - Train: {tw_train:.4f}, Test: {tw_test:.4f}")


# You can pass any embedding function you want. For example PaCMAP or PCA:
def pacmap_embedding(X_train, X_test):
    pacmap_reducer = PaCMAP(random_state=42, n_components=2)
    X_train_embedded = pacmap_reducer.fit_transform(X_train)
    X_test_embedded  = pacmap_reducer.transform(X_test, X_train)
    return X_train_embedded, X_test_embedded

def pca_embedding(X_train, X_test):
    # Initialize PCA and fit/transform on the training data
    pca = PCA(n_components=2)
    X_train_embedded = pca.fit_transform(X_train)
    # Transform the test data using the same PCA model
    X_test_embedded = pca.transform(X_test)
    return X_train_embedded, X_test_embedded


