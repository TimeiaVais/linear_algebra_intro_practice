import numpy as np
from scipy.linalg import lu

def lu_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The permutation matrix P, lower triangular matrix L, and upper triangular matrix U.
    """
    P, L, U = lu(x)
    return P, L, U


def qr_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform QR decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray]: The orthogonal matrix Q and upper triangular matrix R.
    """
    Q, R = np.linalg.qr(x)
    return Q, R


def determinant(x: np.ndarray) -> np.ndarray:
    """
    Calculate the determinant of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The determinant of the matrix.
    """
    return np.linalg.det(x)

def eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and right eigenvectors of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The eigenvalues and the right eigenvectors of the matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(x)
    return eigenvalues, eigenvectors

def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices U, S, and V.
    """
    U, S, Vh = np.linalg.svd(x, full_matrices=True)
    return U, S, Vh


# ================== EXAMPLES / TESTS ==================
if __name__ == "__main__":
    A = np.array([[4, 3], [6, 3]], dtype=float)
    print("Matrix A:\n", A)

    # LU decomposition
    P, L, U = lu_decomposition(A)
    print("\nLU Decomposition:")
    print("P:\n", P)
    print("L:\n", L)
    print("U:\n", U)

    # QR decomposition
    Q, R = qr_decomposition(A)
    print("\nQR Decomposition:")
    print("Q:\n", Q)
    print("R:\n", R)

    # Determinant
    det_A = determinant(A)
    print("\nDeterminant of A:", det_A)

    # Eigenvalues and eigenvectors
    values, vectors = eigen(A)
    print("\nEigenvalues:\n", values)
    print("Eigenvectors:\n", vectors)

    # Singular Value Decomposition
    U_svd, S_svd, Vh_svd = svd(A)
    print("\nSVD Decomposition:")
    print("U:\n", U_svd)
    print("Singular values:\n", S_svd)
    print("Vh:\n", Vh_svd)

# ================== OUTPUT ==================

Matrix A:
 [[4. 3.]
 [6. 3.]]

LU Decomposition:
P:
 [[0. 1.]
 [1. 0.]]
L:
 [[1.         0.        ]
 [0.66666667 1.        ]]
U:
 [[6. 3.]
 [0. 1.]]

QR Decomposition:
Q:
 [[-0.5547002  -0.83205029]
 [-0.83205029  0.5547002 ]]
R:
 [[-7.21110255 -4.16025147]
 [ 0.         -0.83205029]]

Determinant of A: -6.0

Eigenvalues:
 [ 7.77200187 -0.77200187]
Eigenvectors:
 [[ 0.62246561 -0.53222953]
 [ 0.78264715  0.8466001 ]]

SVD Decomposition:
U:
 [[-0.59581566 -0.80312122]
 [-0.80312122  0.59581566]]
Singular values:
 [8.33557912 0.71980602]
Vh:
 [[-0.86400595 -0.50348159]
 [ 0.50348159 -0.86400595]]
