# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from basenmf import NMF, NMFresult
from initialization import rnd_init, rndc_init, rnda_init, svd_init


def initialize_H(X, rank, init):
    if init == "rnd":
        H = rnd_init(X, rank)
    elif init == "rndc":
        H = rndc_init(X, rank)
    elif init == "rnda":
        H = rnda_init(X, rank)
    elif init == "svd":
        H = svd_init(X, rank, flag=1)
    else:
        raise ValueError(f"Unknown initialization: {init}")

    return np.asarray(H, dtype=float)


class SNMTF(NMF):
    """
    Symmetric nonnegative matrix trifactorization:
        X ≈ H @ S @ H.T

    X is sparse; H and S are dense.
    """

    def fit(self):
        X = sparse.csr_matrix(self.X, dtype=float, copy=True)
        X.sum_duplicates()

        if np.any(X.data < 0):
            raise ValueError("Negative entries are not allowed!")

        H = initialize_H(X, self.rank, self.init)
        eps = 1e-10

        # Initialize S and adjust it with H fixed.
        S = np.ones((self.rank, self.rank))

        G = H.T @ H
        B = H.T @ (X @ H)

        for _ in range(20):
            S *= B / np.maximum(G @ S @ G, eps)

        # Squared Frobenius norm of sparse X.
        normX2 = np.dot(X.data, X.data)

        def objective():
            G = H.T @ H
            B = H.T @ (X @ H)

            # ||X - H @ S @ H.T||_F²
            # without forming a dense n × n matrix.
            value = (
                normX2
                - 2 * np.sum(S * B)
                + np.trace(S.T @ G @ S @ G)
            )
            return max(float(value), 0.0)

        previous = objective()
        objfun_vals = []
        converged = False

        for it in range(self.maxiter):
            # Update H.
            numerator = (X @ H) @ S
            denominator = H @ (H.T @ numerator)

            H *= numerator / np.maximum(denominator, eps)

            # Update S using the updated H.
            G = H.T @ H
            B = H.T @ (X @ H)

            S *= B / np.maximum(G @ S @ G, eps)

            if it % 10 == 0 or it == self.maxiter - 1:
                value = objective()
                objfun_vals.append(value)
                decrease = previous - value

                if self.displ:
                    print(
                        f"### Iter = {it} | ObjF = {value:.3e}"
                        f" | Decrease = {decrease:.3e}"
                    )

                if 0 <= decrease < self.tol:
                    converged = True
                    break

                previous = value
        norm_X = np.sqrt(X.multiply(X).sum())

        # Identité : ||H H.T||_F = ||H.T H||_F
        norm_HHt = np.linalg.norm(H.T @ H, "fro")

        if norm_X > 0 and norm_HHt > 0:
            scale = np.sqrt(norm_X / norm_HHt)

            H = scale * H
            S = S / scale**2

        return NMFresult(
            (H, S),
            np.asarray(objfun_vals),
            objective(),
            converged
        )