# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from basenmf import NMF, NMFresult
from initialization import rnd_init, rndc_init, rnda_init, svd_init


class SNMF(NMF):
    """
    Symmetric nonnegative matrix factorization:
        X ≈ H @ H.T

    X is sparse; H is dense.
    """

    def fit(self):
        X = sparse.csr_matrix(self.X, dtype=float, copy=True)
        X.sum_duplicates()

        if np.any(X.data < 0):
            raise ValueError("Negative entries are not allowed!")

        if self.init == "rnd":
            H = rnd_init(X, self.rank)
        elif self.init == "rndc":
            H = rndc_init(X, self.rank)
        elif self.init == "rnda":
            H = rnda_init(X, self.rank)
        elif self.init == "svd":
            H = svd_init(X, self.rank, flag=1)
        else:
            raise ValueError(f"Unknown initialization: {self.init}")

        H = np.asarray(H, dtype=float)

        normX2 = np.dot(X.data, X.data)

        def reconstruction_error():
            XH = X @ H
            G = H.T @ H

            # ||X - H @ H.T||_F² without forming H @ H.T.
            error2 = (
                normX2
                - 2 * np.sum(H * XH)
                + np.sum(G * G)
            )
            return np.sqrt(max(float(error2), 0.0))

        pdist = 1e10
        converged = False
        objfun_vals = []

        for it in range(self.maxiter):
            # Multiplicative update
            grad_neg = X @ H
            grad_pos = H @ (H.T @ H)

            H *= 0.5 + 0.5 * grad_neg / (grad_pos + 1e-10)

            if it % 10 == 0 or it == self.maxiter - 1:
                dist = reconstruction_error()
                objfun_vals.append(dist * dist)
                decrease = pdist - dist

                if self.displ:
                    print(
                        f"### Iter = {it} | ObjF = {dist * dist:.3e}"
                        f" | Decrease = {decrease:.3e}"
                    )

                if decrease < self.tol:
                    converged = True
                    break

                pdist = dist

        # Final error before row normalization
        dist = reconstruction_error()

        # L1 row normalization
        norms = H.sum(axis=1)
        norms[norms == 0] = 1.0
        H /= norms[:, np.newaxis]

        return NMFresult(
            (H,),
            np.asarray(objfun_vals),
            dist * dist,
            converged
        )