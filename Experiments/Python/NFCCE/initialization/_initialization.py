# -*- coding: utf-8 -*-

import numpy as np
from math import ceil, sqrt
from numpy import linalg as la
from scipy import sparse
from scipy.sparse.linalg import svds


def rnd_init(X, k):
    """
    Random initialization.
    Returns a dense ndarray.
    """
    return np.random.rand(X.shape[0], k)


def rnda_init(X, k, p=None):
    """
    RandomAcol initialization.
    Supports sparse and dense X.
    """
    if sparse.issparse(X):
        X = sparse.csc_matrix(X, dtype=float)
    else:
        X = np.asarray(X, dtype=float)

    if p is None:
        p = int(ceil(X.shape[1] / 5))

    prng = np.random
    H = np.zeros((X.shape[0], k))

    for i in range(k):
        indices = prng.randint(
            low=0, high=X.shape[1], size=p
        )
        H[:, i] = np.asarray(
            X[:, indices].mean(axis=1)
        ).ravel()

    return H


def rndc_init(X, k, p=None, l=None):
    """
    RandomC initialization.
    Preserves the original row-norm ranking for square X.
    Supports sparse and dense X.
    """
    if X.shape[0] != X.shape[1]:
        raise ValueError("rndc_init expects a square matrix.")

    if sparse.issparse(X):
        X = sparse.csc_matrix(X, dtype=float)
        row_norms = np.sqrt(
            np.asarray(X.multiply(X).sum(axis=1)).ravel()
        )
    else:
        X = np.asarray(X, dtype=float)
        row_norms = la.norm(X, axis=1)

    if p is None:
        p = int(ceil(X.shape[0] / 5))

    if l is None:
        l = int(ceil(X.shape[0] / 2))

    prng = np.random
    H = np.zeros((X.shape[0], k))

    # Stable sorting preserves the original ordering for ties.
    top = np.argsort(-row_norms, kind="stable")[:l]

    for i in range(k):
        indices = top[
            prng.randint(low=0, high=len(top), size=p)
        ]
        H[:, i] = np.asarray(
            X[:, indices].mean(axis=1)
        ).ravel()

    return H


def svd_init(X, k, flag=0):
    """
    SVD-based initialization.

    Preserves the original positive-part construction.
    Uses a truncated SVD for sparse X, without densifying X.

    Reference:
        Boutsidis, C., & Gallopoulos, E. (2008).
        "SVD based initialization: A head start for
        nonnegative matrix factorization."
        Pattern Recognition, 41(4), 1350-1362.
    """
    is_sparse = sparse.issparse(X)

    if is_sparse:
        X = sparse.csr_matrix(X, dtype=float)

        if not 1 <= k < min(X.shape):
            raise ValueError(
                "For sparse SVD, k must satisfy "
                "1 <= k < min(X.shape)."
            )
    else:
        X = np.asarray(X, dtype=float)

        if not 1 <= k <= min(X.shape):
            raise ValueError(
                "k must satisfy 1 <= k <= min(X.shape)."
            )

    H = np.zeros((X.shape[0], k))

    # Handle the zero matrix without calling the SVD solver.
    is_zero = (
        np.count_nonzero(X.data) == 0
        if is_sparse
        else not np.any(X)
    )
    if is_zero:
        return H

    if is_sparse:
        U, S, _ = svds(X, k=k, which="LM")

        # svds does not guarantee descending order.
        order = np.argsort(S)[::-1]
        U = U[:, order]
        S = S[order]
    else:
        U, S, _ = la.svd(X, full_matrices=False)

    H[:, 0] = sqrt(S[0]) * np.abs(U[:, 0])

    for i in range(1, k):
        uup = _pos(U[:, i])
        n_uup = la.norm(uup, 2)

        if n_uup > 0:
            termp = n_uup * n_uup
            H[:, i] = sqrt(S[i] * termp) / n_uup * uup

    H[H < 1e-10] = 0

    if flag == 1:
        H[H == 0] = float(X.mean())

    elif flag == 2:
        avg = float(X.mean())
        mask = H == 0
        H[mask] = (
            avg
            * np.random.uniform(0.0, 1.0, size=np.count_nonzero(mask))
            / 100.0
        )

    return H


def _pos(X):
    """
    Return positive elements of X.
    """
    return np.maximum(X, 0)


def _neg(X):
    """
    Return absolute values of negative elements of X.
    """
    return np.maximum(-X, 0)