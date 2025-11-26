from pacmap import PaCMAP
from sklearn.decomposition import PCA
import umap


def pacmap_embedding(X_train, X_test, dimentions):
    pacmap_reducer = PaCMAP(random_state=42, n_components=dimentions)
    X_train_embedded = pacmap_reducer.fit_transform(X_train)
    X_test_embedded  = pacmap_reducer.transform(X_test, X_train)
    return X_train_embedded, X_test_embedded

def pca_embedding(X_train, X_test, dimentions):
    # Initialize PCA and fit/transform on the training data
    pca = PCA(n_components=dimentions)
    X_train_embedded = pca.fit_transform(X_train)
    # Transform the test data using the same PCA model
    X_test_embedded = pca.transform(X_test)
    return X_train_embedded, X_test_embedded


def umap_embedding(X_train, X_test, dimensions):
    umap_reducer = umap.UMAP(n_components=dimensions, random_state=42)
    X_train_embedded = umap_reducer.fit_transform(X_train)
    X_test_embedded = umap_reducer.transform(X_test)
    return X_train_embedded, X_test_embedded