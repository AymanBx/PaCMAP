from sklearn.manifold import trustworthiness


def run_trustworthiness(X_train_flat, X_train_embedded, X_test_flat, X_test_embedded, embedding_func, n_neighbors_list=[1, 3, 5, 7, 9]):
    """
    Computes trustworthiness between original data and reduced embeddings.
    """

    # Evaluate trustworthiness
    for k in n_neighbors_list:
        tw_train = trustworthiness(X_train_flat, X_train_embedded, n_neighbors=k)
        tw_test = trustworthiness(X_test_flat, X_test_embedded, n_neighbors=k)
        print(f"Trustworthiness (k={k}) - Train: {tw_train:.4f}, Test: {tw_test:.4f}")