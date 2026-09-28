# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from basenmf import NMF, NMFresult
from snmtf import SNMTF, initialize_H


class CSNMTF(NMF):
    """
    Collective SNMTF with sparse input matrices.

    First factorize each layer into H_local[i] and S[i].
    Then update the common H with these local factors fixed.
    """

    def __init__(self, X, rank, alpha=0.5, numRestarts=3, **kwargs):
        NMF.__init__(self, X, rank, **kwargs)
        self.alpha = alpha
        self.numRestarts= numRestarts

    def fit(self):
        X = [
            sparse.csr_matrix(A, dtype=float, copy=True)
            for A in self.X
        ]
        N = len(X)
        eps = 1e-10

        if N == 0:
            raise ValueError("At least one network is required.")

        for A in X:
            A.sum_duplicates()

            if np.any(A.data < 0):
                raise ValueError("Negative entries are not allowed!")
        

        # Step 1: independent layer factorizations. 
        # Select the factorization with the minimal error from 10 restarts 
        Hi_best=[]
        Si_best=[]
        error_best=np.inf
        for restart in range(self.numRestarts):
            
            H_local = []
            S = []
            error=0
            for i in range(N):
                if self.displ:
                    print(f"### Factorizing network [{i + 1}]...")

                model = SNMTF(
                    X[i],
                    self.rank,
                    init=self.init,
                    displ=self.displ,
                    maxiter=self.maxiter,
                    tol=self.tol
                )
                result = model.fit()

                H_local.append(
                    np.asarray(result.matrices[0], dtype=float).copy()
                )
                S.append(
                    np.asarray(result.matrices[1], dtype=float).copy()
                )
                error+=result.objfun_final
            if error < error_best:
                error_best=error
                Hi_best=[Hi.copy() for Hi in H_local]
                Si_best=[Si.copy() for Si in S]
                

                

        # Step 2: initialize the common factor.
        # The sparse mean is used only for initialization.
        # Collective factorization , select the restart with the minimum reconstruction error 
        
        #Precomputation 
        X_avg = sparse.csr_matrix(X[0].shape, dtype=float)

        for A in X:
            X_avg = X_avg + A

        X_avg /= N

        H_best=[]
        S_best=[]
        error_best=np.inf
        objfun_vals_best=[]
        converged=False 
        normX2 = [np.dot(A.data, A.data) for A in X]
        
        def objective():
                    G = H.T @ H
                    value = 0.0
        
                    for i in range(N):
                        B = H.T @ (X[i] @ H)
        
                        # Reconstruction error without forming H @ S[i] @ H.T.
                        reconstruction = (
                            normX2[i]
                            - 2 * np.sum(S[i] * B)
                            + np.trace(S[i].T @ G @ S[i] @ G)
                        )
        
                        overlap = Hi_best[i].T @ H
        
                        # Grassmann coupling, omitting alpha * N * rank.
                        value += reconstruction
                        value -= self.alpha * np.sum(overlap * overlap)
        
                    return float(value)
        for restart in range(self.numRestarts):
            H = initialize_H(X_avg, self.rank, self.init)
            S=[Si.copy() for Si in Si_best]
            # Squared Frobenius norms of the sparse input matrices.
            

            

            previous = objective()
            objfun_vals = []
            converged = False
            

            for it in range(self.maxiter):
                # Update of H 
                numerator = np.zeros_like(H)
                for i in range(N):
                    numerator += (X[i] @ H) @ S[i]
                    numerator += (self.alpha / 2) * (
                        Hi_best[i] @ (Hi_best[i].T @ H)
                    )
               
                denominator = H @ (H.T @ numerator)
                H *= numerator / np.maximum(denominator, eps)

                #Update of S 
                G = H.T @ H

                for i in range(N):
                    B = H.T @ (X[i] @ H)
                    S[i] *= B / np.maximum(G @ S[i] @ G, eps)


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
            if value <error_best:
                 error_best= value
                 H_best= H.copy()
                 S_best=[Si.copy() for Si in S]
                 objfun_vals_best= objfun_vals.copy()
                 converged_best=converged
        
           

        return NMFresult(
            (H_best, S_best),
            np.asarray(objfun_vals_best),
            error_best,
            converged_best,
        )