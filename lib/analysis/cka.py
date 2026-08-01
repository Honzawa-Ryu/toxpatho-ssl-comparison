# -*- coding: utf-8 -*-
"""
Centered Kernel Alignment (CKA) for comparing representations across models.

CKA is invariant to orthogonal transforms and isotropic scaling, and (unlike
plain correlation) works across representations of *different dimensionality*
-- which is exactly what we need to compare a 2048-d ResNet space against a
768-d ViT space.

Reference:
    Kornblith, Norouzi, Lee, Hinton (2019),
    "Similarity of Neural Network Representations Revisited", ICML.

@author: wsi-ad
"""
import numpy as np


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix: H K H, H = I - 1/n."""
    n = K.shape[0]
    unit = np.ones((n, n), dtype=K.dtype) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """(Biased) HSIC estimator on centered Gram matrices."""
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    return float(np.sum(Kc * Lc))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between feature matrices X (n, p) and Y (n, q).

    Uses the efficient linear-kernel form; equivalent to CKA with K=XX^T.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # ||Y^T X||_F^2
    cross = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")
    denom = norm_x * norm_y
    if denom == 0:
        return float("nan")
    return float(cross / denom)


def _rbf_gram(X: np.ndarray, sigma_frac: float = 0.5) -> np.ndarray:
    """RBF Gram matrix with bandwidth set from the median pairwise distance."""
    sq = np.sum(X ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * (X @ X.T)
    np.maximum(d2, 0, out=d2)
    # median heuristic on the off-diagonal distances
    tri = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(tri) if tri.size else 1.0
    sigma2 = sigma_frac * med + 1e-12
    return np.exp(-d2 / (2.0 * sigma2))


def kernel_cka(X: np.ndarray, Y: np.ndarray, sigma_frac: float = 0.5) -> float:
    """RBF-kernel CKA between X (n, p) and Y (n, q)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    K = _rbf_gram(X, sigma_frac)
    L = _rbf_gram(Y, sigma_frac)
    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom == 0:
        return float("nan")
    return float(hsic_kl / denom)


def cka_matrix(embeddings: dict, kernel: str = "linear", sigma_frac: float = 0.5) -> tuple:
    """Pairwise CKA matrix over a dict {name: (n, d) array}.

    All arrays must share the same n (same samples, same order).

    Returns (names, matrix) where matrix[i, j] = CKA(embeddings[i], embeddings[j]).
    """
    names = list(embeddings.keys())
    m = len(names)
    mat = np.eye(m, dtype=np.float64)
    fn = linear_cka if kernel == "linear" else (
        lambda a, b: kernel_cka(a, b, sigma_frac=sigma_frac))
    for i in range(m):
        for j in range(i, m):
            val = fn(embeddings[names[i]], embeddings[names[j]])
            mat[i, j] = val
            mat[j, i] = val
    return names, mat
