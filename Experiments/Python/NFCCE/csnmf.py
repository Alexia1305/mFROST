# -*- coding: utf-8 -*-

from scipy import sparse

from snmf import SNMF
from basenmf import NMF


class CSNMF(NMF):
    """
    Implementation of the Collective SNMF (CSNMF).
    """

    def __init__(self, X, rank, alpha=0.5, **kwargs):
        NMF.__init__(self, X, rank, **kwargs)
        self.alpha = alpha

    def fit(self):
        N = len(self.X)

        if N == 0:
            raise ValueError("At least one network is required.")

        A_avg = sparse.csr_matrix(self.X[0].shape, dtype=float)

        for i in range(N):
            if self.displ:
                print(f"### Factorizing network [{i + 1}]...")

            X = sparse.csr_matrix(
                self.X[i], dtype=float, copy=True
            )
            X.sum_duplicates()

            objSNMF = SNMF(
                X,
                self.rank,
                init=self.init,
                displ=self.displ,
                maxiter=self.maxiter,
                tol=self.tol
            )
            res_snmf = objSNMF.fit()

            Hi = sparse.csr_matrix(res_snmf.matrices[0])
            Hi.eliminate_zeros()

            A_avg = A_avg + X + self.alpha * (Hi @ Hi.T)

            if self.displ and res_snmf.converged:
                print("### Converged.")

        A_avg /= (1.0 + self.alpha) * N
        A_avg.eliminate_zeros()

        objSNMF = SNMF(
            A_avg,
            self.rank,
            init=self.init,
            displ=self.displ,
            maxiter=self.maxiter,
            tol=self.tol
        )

        return objSNMF.fit()