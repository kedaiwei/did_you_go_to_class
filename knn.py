import numpy as np
from scipy.spatial.distance import cdist


def flatten_normalize_matrices(matrices):
    """
    Flattens and normalizes a list of 2D matrices into a 2D array where each row is a flattened matrix.
    """
    matrix = np.array([m.flatten() for m in matrices])
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def mutual_knn(mat1, mat2):
    """
    Returns the indices from matrix 1 that have mutual nearest neighbors in matrix 2
    """
    mat1_flat = flatten_normalize_matrices(mat1)
    mat2_flat = flatten_normalize_matrices(mat2)

    dist_matrix = cdist(mat1_flat, mat2_flat)

    nn_mat1_to_mat2 = np.argmin(dist_matrix, axis=1)

    nn_mat2_to_mat1 = np.argmin(dist_matrix, axis=0)

    mutual_nn_indices = [
        i for i in range(len(mat1)) if nn_mat2_to_mat1[nn_mat1_to_mat2[i]] == i
    ]

    return mutual_nn_indices


# Example
mat1 = np.array([
    [[1, 2],  # Alex; feature 1
     [3, 4]],  # Alex; feature 2

    [[5, 6],  # Bob; feature 1
     [7, 8]]  # Bob; feature 2
])
mat2 = np.array([
    [[1, 2],  # Alex; feature 1
     [3, 4.1]],  # Alex; feature 2

    [[1.2, 2.2],  # Carl; feature 1
     [3.2, 4.2]]  # Carl; feature 2
])

print(mutual_knn(mat1, mat2))


#######################################


def normalize_rows(matrix):
    """Normalizes each row of a 2D matrix using L2 norm."""
    return matrix
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def row_mutual_knn(mat1, mat2):
    num_samples1, _, _ = np.shape(mat1)
    num_samples2, _, _ = np.shape(mat2)

    mat1 = np.array([normalize_rows(m) for m in mat1])
    mat2 = np.array([normalize_rows(m) for m in mat2])

    # mat1 --> mat2
    best_match_mat1 = []

    for i in range(num_samples1):
        min_distances = np.inf
        best_index = -1

        for j in range(num_samples2):
            row_dists = cdist(mat1[i], mat2[j])
            avg_dist = np.mean(np.min(row_dists, axis=1))

            if avg_dist < min_distances:
                min_distances = avg_dist
                best_index = j

        best_match_mat1.append(best_index)

    # mat2 --> mat1
    best_match_mat2 = []

    for j in range(num_samples2):
        min_distances = np.inf
        best_index = -1

        for i in range(num_samples1):
            row_dists = cdist(mat2[j], mat1[i])
            avg_dist = np.mean(np.min(row_dists, axis=1))

            if avg_dist < min_distances:
                min_distances = avg_dist
                best_index = i

        best_match_mat2.append(best_index)

    # Find indices in mat1 where the nearest neighbor relationship is mutual
    mutual_nn_indices = [i for i in range(
        num_samples1) if best_match_mat2[best_match_mat1[i]] == i]

    return mutual_nn_indices


# Example
mat1 = np.array([
    [[1, 2],  # Alex; feature 1
     [3, 4,]],  # Alex; feature 2

    [[5, 6],  # Bob; feature 1
     [7, 8]],  # Bob; feature 2

    [[9, 8],  # Dan; feature 1
     [7, 6]]  # Dan; feature 2
])
mat2 = np.array([
    [[9.1, 8.2],  # Dan; feature 1
     [7.1, 6.3]],  # Dan; feature 2

    [[1, 2],  # Alex; feature 1
     [3, 4.1]],  # Alex; feature 2

    [[1.2, 2.2],  # Carl; feature 1
     [3.2, 4.2]]  # Carl; feature 2
])

print(row_mutual_knn(mat1, mat2))
