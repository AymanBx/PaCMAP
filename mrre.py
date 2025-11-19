import numpy as np
from sklearn.metrics import pairwise_distances


def compute_mrre(X_high, X_low, K=10):
    """
    X_high: data in high-dimensional space
    X_low: embedding in low-dimensional space
    K: neighborhood size
    """

    # Compute pairwise distances
    # D_high[i, j] = distance between point i and j in original space
    D_high = pairwise_distances(X_high)
    D_low = pairwise_distances(X_low)

    n = X_high.shape[0]

    # Get ranking of neighbors (argsort returns indices in sorted order)
    # If point 7’s closest points (in order) are [7, 42, 12, 9, 3],
    # ranks_high[7] == [7, 42, 12, 9, 3, ...].
    ranks_high = np.argsort(D_high, axis=1)
    ranks_low = np.argsort(D_low, axis=1)

    mrre_total = 0.0

    for i in range(n):
        # Skip self (rank 0)
        high_neighbors = ranks_high[i][1:K+1]

        # For each true neighbor j, get its rank in low dimension
        for j in high_neighbors:
            rank_low = np.where(ranks_low[i] == j)[0][0]

            ideal_rank = np.where(high_neighbors == j)[0][0] + 1

            rel_error = abs(rank_low - ideal_rank) / ideal_rank
            mrre_total += rel_error

    mrre = mrre_total / (n * K)
    return mrre


def run_mrre(X_high, X_low):
    """
    Compute MRRE for multiple K values: 1, 3, 5, 7, 9
    X_high: original high-dimensional data
    X_low: low-dimensional embedding
    """

    K_values = [1, 3, 5, 7, 9]
    print("\n=== MRRE Evaluation ===")

    for K in K_values:
        mrre = compute_mrre(X_high, X_low, K=K)
        print(f"MRRE (K={K}): {mrre:.4f}")
