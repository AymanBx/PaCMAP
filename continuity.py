import numpy as np
from sklearn.metrics import pairwise_distances


def compute_continuity(X_high, X_low, K=10):
    D_high = pairwise_distances(X_high)
    D_low = pairwise_distances(X_low)

    n = X_high.shape[0]

    ranks_high = np.argsort(D_high, axis=1)
    ranks_low = np.argsort(D_low, axis=1)

    continuity_sum = 0.0
    
    for i in range(n):
        high_neighbors = ranks_high[i][1:K+1]
        low_neighbors = set(ranks_low[i][1:K+1])

        # neighbors that SHOULD be present but are missing
        U = [j for j in high_neighbors if j not in low_neighbors]

        for j in U:
            # rank in LOW-D space (this is the correct part)
            r_lo = np.where(ranks_low[i] == j)[0][0]
            continuity_sum += (r_lo - K)

    # correct normalization for continuity
    normalization = n * K * (2 * n - K - 1)

    C = 1 - (2 * continuity_sum) / normalization
    return max(min(C, 1.0), 0.0)  # guard against floating drift


def run_continuity(X_high, X_low):
    """
    Compute Continuity for multiple K values: 1, 3, 5, 7, 9
    """

    K_values = [1, 3, 5, 7, 9]
    print("\n=== Continuity Evaluation ===")

    for K in K_values:
        cont = compute_continuity(X_high, X_low, K=K)
        print(f"Continuity (K={K}): {cont:.4f}")
