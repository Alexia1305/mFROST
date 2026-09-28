
import time
from scipy.sparse import issparse, find, csr_matrix
from scipy.sparse.linalg import norm
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score

from mfrost.consensus import MatrixFreeSpectralClusteringCoAssociation, USENC_ConsensusFunction
import numpy as np
import math

# math.cbrt is available from Python 3.11; keep older Python compatible.
_scalar_cbrt = getattr(math, "cbrt", np.cbrt)
from .SVCA import svca
from scipy.sparse import diags

# -----------------------------------------------
# ---------------- frost ----------------
# -----------------------------------------------

def mfrost(X_list, r, numTrials=10, maxiter=50, delta=1e-5, time_limit=None, init_method='USENC',
                     init_partition=None, verbosity=0, init_seed=None):
    """
    Heuristic algorithm for multilayer community detection via joint nonnegative matrix trifactorization.
    Estimates nonnegative matrices S_l>=0 and Z_l>=0 that minimize:

        sum_{l=1..L} ||X_l-Z_l S_l Z_l^T||_F^T

    subject to the constraints:
        Z_l = D_l V
        Z_l^T Z_l = I
        S_l >= 0
        Z_l >= 0
    where:
    - X_l is the adjacency matrix of layer l
    - S_l is a nonnegative community interaction matrix for layer l
    - Z_l is a nonnegative orthogonal matrix encoding community memberships
    - D_l is a diagonal scaling matrix specific to layer l
    - V is a shared binary community assignment matrix across all layers

    Notes
    -----
    The matrices Z_l=D_l V are not stored explicitly but represented with:

    - v : ndarray of shape (n,) — layer-independent community assignments defining V,
      where v[i] is the community index of node i.

    - w : ndarray of shape (L, n) — layer-dependent diagonal values defining D_l,
      where w[l, i] is the scaling factor for node i in layer l.

    Thus, each row i of Z_l has a single nonzero element:

        Z_l[i, v[i]] = w[l, i]

    The vector v is shared across layers, while w is layer-dependent.

     Parameters:
         X_list : list of crs_matrix
             List of L matrices, one per layer.
             - Each matrix has shape (n, n), where n is the number of nodes.
             - Matrices are nonnegative and symmetric for adjacency matrix of an undirected graphs.
         r : int
             Number of communities.
         numTrials : int, default=10
             Number of restarts with different initializations.
         maxiter : int, default=50
             Maximum iterations for each trial.
         delta : float, default=1e-6
             Convergence tolerance (Stop if error_prec-error<delta).
         time_limit : int, default=None
             Time limit in seconds.
         init_method : str, default = USENC
             Initialization method ("random", "USENC").
         verbosity : int, default=1
             (1 for messages, 0 for silent mode).
         init_partition : np.array, shape (n,), default=None
            Initial node partition 
         init_seed : float, default=None
             Random seed for the initialization for the experiments


     Returns:
         w_best : ndarray of shape (L, n)
            Layer-specific nonzero values of Z_l (diagonal elements of D_l). w_best[i,l] is the sociability of the node i in the layer l
         v_best : np.array, shape (n,)
             Shared community assignment vector. v_best[i] is the community of node i.
         S_best : ndarray of shape (L, r, r)
             Layer-specific community interaction matrices S_l.
         error_best : float
             Relative error sqrt( sum_l ||X_l - Z_l S_l Z_l'||^2_F / (sum_l ||X_l||_F^2)).
    
     """
    
    convergence_data = []
    start_time = time.time()

    # List of adjacency matrices 
    if (isinstance(X_list, np.ndarray) or issparse(X_list)) and X_list.ndim == 2 and X_list.shape[0] == X_list.shape[1]:
        X_list = [X_list]

    if not isinstance(X_list, list):
        raise TypeError("Xlist must be a Python list")

    X_list = [csr_matrix(X).astype('float') if not issparse(X) else X.tocsr().astype('float') for X in X_list]

   # Instances initialization
    L = len(X_list)
    n = X_list[0].shape[0]
    error_best = float('inf')

    w_best = np.zeros((L, n))
    S_best = np.zeros((L, r, r))
    v_best = np.zeros(n)

    # Precomputations
    normX = [norm(X, 'fro') for X in X_list]
   
    degrees_layers = []

    for X in X_list:
        degrees = np.asarray(X.sum(axis=1)).ravel()
        degrees_layers.append(degrees)

    # X never changes: reuse its rows, diagonal and edge list 
    X_cache = _prepare_X_cache(X_list)

    if verbosity > 0:
        print(f'Running {numTrials} Trials in Series')

    # Restarts with different initialization   

    for trial in range(numTrials):
        
        # Initialization of Z_l stored by v and w
        if init_partition is not None:
            v = np.asarray(init_partition)
            if (v.shape != (n,) or not np.issubdtype(v.dtype, np.integer)
                    or np.any(v < 0) or np.any(v >= r)):
                raise ValueError("init_partition must contain n integer labels in [0, r).")
            v = v.copy()
            w = np.zeros((L, n))
            for l in range(L):
                w[l] = initialize_w_values(X_list[l], v)

        else:

            if L == 1 :
                init_method = 'onelayer'

            if init_seed is not None:
                init_seed += 10 * trial
                np.random.seed(init_seed)

            if init_method == 'random':
                base = np.arange(r)
                rest = np.random.randint(0, r, size=n - r)
                v = np.concatenate([base, rest])
                np.random.shuffle(v)
                w = np.zeros((L, n))
                for l in range(L):
                    w[l] = initialize_w_values(X_list[l], v)
               
            elif init_method == 'onelayer':
                w = np.zeros((L, n))
                lr = np.random.randint(0, L)  # Choose a random layer
                w[lr], v, _ = initialize_Z_onelayer(X_list[lr], r)
                # default value degree of the node / sum degrees same community
                for l in range(L):
                    if l == lr:
                        continue
                    w[l] = initialize_w_values(X_list[l], v)
                

            else:

                w, v = initialize_Z_alllayers(X_list, r, init_method)

        # Normalization of Z (w)
        _normalize_w(w, v, r)

        # Compute of S
        S = update_S(X_list, r, w, v, X_cache=X_cache)

        if verbosity:
            print('Time', time.time() - start_time)
        prev_error = 0
        for l in range(L):
            prev_error += compute_error(normX[l], S[l])**2
        prev_error=np.sqrt(prev_error/sum(x ** 2 for x in normX))
        error = prev_error
        
        for iteration in range(maxiter):
           
            if time_limit and time.time() - start_time > time_limit:
                print('Time limit passed')
                break
            
           
            w, v = update_Z(X_list, degrees_layers, S, w, v, X_cache=X_cache)

            S = update_S(X_list, r, w, v, X_cache=X_cache)

            prev_error = error
            error = 0
            for l in range(L):
                error += compute_error(normX[l], S[l])**2
            error=np.sqrt(error/sum(x ** 2 for x in normX))

            if delta:
                if error < delta or abs(prev_error - error) < delta:
                    break

        if error < error_best:
            w_best, v_best, S_best, error_best = (
                 w.copy(), v.copy(), S.copy(), error
            )
       
        if verbosity > 0:
            print(f'Trial {trial + 1}/{numTrials} with {init_method}: Error {error:.4e} | Best: {error_best:.4e}')
            print('Time', time.time() - start_time)
            
        if time_limit and time.time() - start_time > time_limit:
                print('Time limit passed')
                break
    
    return w_best, v_best, S_best, error_best
def _prepare_X_cache(X_list):
    """
    rows[i][l] contains the column indices and values of X_l[i, :],
    excluding the diagonal. For rows without a diagonal entry these are
    views into the CSR arrays, with no copy of the numerical data.
    The cache uses O(sum_l nnz(X_l) + L*n) extra storage, 
    """
    n = X_list[0].shape[0]
    rows = [[] for _ in range(n)]
    diagonal = np.empty((len(X_list), n))
    edges = []
    for l, X in enumerate(X_list):
        X = X.tocsr()
        diagonal[l] = X.diagonal()
        edges.append(find(X))
        for i in range(n):
            start, end = X.indptr[i:i + 2]
            cols = X.indices[start:end]
            vals = X.data[start:end]
            keep = cols != i
            if not np.all(keep):
                cols, vals = cols[keep], vals[keep]
            rows[i].append((cols, vals))
    return {"rows": rows, "diagonal": diagonal, "edges": edges}


def _normalize_w(w, v, r):
    """ Normalization of w (ZTZ=I)"""
    for wl in w:
        nw = np.sqrt(np.bincount(v, weights=wl**2, minlength=r))
        denom = nw[v]
        nonzero = denom != 0
        np.divide(wl, denom, out=wl, where=nonzero)
        wl[~nonzero] = 0



def update_Z(X_list, degrees_layers, S, w, v, X_cache=None):
    """Update of Z """
    L = len(X_list)
    n = X_list[0].shape[0]
    r = S.shape[1]
    if X_cache is None:
        X_cache = _prepare_X_cache(X_list)
    rows = X_cache["rows"]
    Xii = X_cache["diagonal"]

    # Sum w_j^2 once per community, then multiply by S^2. 
    S2 = S**2
    community_norm2 = np.array([
        np.bincount(v, weights=wl**2, minlength=r) for wl in w
    ])
    wp2 = np.einsum("lj,ljk->lk", community_norm2, S2)
    diagS = np.diagonal(S, axis1=1, axis2=2)
    cubic_a = 4 * np.diagonal(S2, axis1=1, axis2=2)
    has_degree = np.asarray(degrees_layers) != 0
    c0_all_layers = np.empty((L, r))
    wi = np.empty(L)

    # Update of each row (node)
    for i in np.random.permutation(n):
        vi_new = -1
        wi_new = np.empty(L)
        f_new = np.inf
        old_contribution = (w[:, i, None] * S[:, v[i], :]) ** 2
        c1_all = 2 * (wp2 - old_contribution) - 2 * diagS * Xii[:, i, None]

        
        for l, (cols, vals) in enumerate(rows[i]):
            weights = vals * w[l, cols]
            c0_all_layers[l] = -4 * (weights @ S[l, v[cols], :])

        # Test each community
        for k in range(r):
            
            erreur = 0
            # For each layer, find the best value for w[i] with v[i] = k
            for l in range(L):
                if not has_degree[l, i]:
                    wi[l] = 0
                else:
                    # Cardano method to find the roots and return best solution >=0 
                    x, min_value = cardan_depressed(
                        cubic_a[l, k], 2 * c1_all[l, k], c0_all_layers[l, k]
                    )

                    wi[l] = x

                    erreur += min_value

            if erreur < f_new:
                f_new = erreur
                wi_new[:] = wi
                vi_new = k

        wp2 = wp2 - old_contribution + (wi_new[:, None] * S[:, vi_new, :]) ** 2

        # Update v and w
        v[i] = vi_new
        w[:, i] = wi_new

    # Normalization of Z (w)
    _normalize_w(w, v, r)

    return w, v


def cardan_depressed(a, c, d, tol=1e-12):
    """Minimize a*x**4/4 + c*x**2/2 + d*x over nonnegative stationary roots.
    """
    a, c, d = float(a), float(c), float(d)
    roots = ()
    if abs(a) < tol:
        if abs(c) > tol:
            roots = (-d / c,)
    else:
        p = c / a
        q = d / a
        Delta = 4 * p**3 + 27 * q**2
        if abs(Delta) < tol:
            if p == 0:
                roots = (_scalar_cbrt(-q),)
            elif abs(p) < tol and abs(q) < tol:
                roots = (0.0,)
            else:
                roots = (3*q/p, -3*q/(2*p))
        elif Delta > 0:
            sqrtD = math.sqrt(Delta / 27)
            roots = (_scalar_cbrt((-q + sqrtD) / 2)
                     + _scalar_cbrt((-q - sqrtD) / 2),)
        else:
            radius = 2 * math.sqrt(-p / 3)
            cos_arg = -q / 2 * math.sqrt(-27 / p**3)
            cos_arg = max(-1.0, min(1.0, cos_arg))
            theta = math.acos(cos_arg) / 3
            roots = (radius * math.cos(theta),
                     radius * math.cos(theta + 2 * math.pi / 3),
                     radius * math.cos(theta + 4 * math.pi / 3))

    x = 0.0
    min_value = 0.0
    for sol in roots:
        if sol > 0:
            value = a/4 * sol**4 + c/2 * sol**2 + d * sol
            if value < min_value:
                x, min_value = sol, value
    return x, min_value


def update_S(X_list, r, w, v, X_cache=None):
    """Compute S= Z_l.T @ X_l @ Z_l """
    L = len(X_list)
    S = np.zeros((L, r, r))
    edges = X_cache["edges"] if X_cache is not None else [find(X) for X in X_list]
    for l in range(L):
        i, j, val = edges[l]
        community_pair = v[i] * r + v[j]
        weights = w[l, i] * w[l, j] * val
        S[l] = np.bincount(
            community_pair, weights=weights, minlength=r*r
        ).reshape(r, r)
    return S



# ------------------------------------------------
# -------------- INITIALIZATION ------------------
# ------------------------------------------------

def initialize_Z_alllayers(X_list, r, init_method):
    L = len(X_list)
    n = X_list[0].shape[0]

    Z = np.zeros((L, n, r))
    w = np.zeros((L, n))
    v_init = np.zeros((L, n))
    # Find w and v for each layer
    for l in range(L):
        w[l], v_init[l], Z[l] = initialize_Z_onelayer(X_list[l], r)

    
    if init_method == 'MF-SC-CA':
        method = MatrixFreeSpectralClusteringCoAssociation(v_init, r)
        labels_final = method.fit_predict()

    elif init_method == 'USENC':
        labels_final = USENC_ConsensusFunction(v_init.T, r)

    for l in range(L):
        w[l] = initialize_w_values(X_list[l], labels_final)


    return w, labels_final

def initialize_Z_onelayer(X, r):
    options = {'average': 1}
    n = X.shape[0]
    p = max(2, math.floor(0.1 * n / r))
    # Estimation of ZS=Z*S with SVCA
    # Attention the rank of X must be greater than r
    ZS, K = svca(X.tocsc(), r, p, options=options)
    norm2x_squared = X.multiply(X).sum(axis=0)  # matrice 1 x n (sparse)
    norm2x_squared = np.array(norm2x_squared).ravel()
    norm2x = np.sqrt(norm2x_squared)
    norm2x_safe = norm2x + 1e-16

    # Normalization of the X columns
    inv_norms = 1.0 / norm2x_safe
    D = diags(inv_norms)
    Xn = X @ D
    # Solve ||X-ZSHO||_F with HOHO^T = D
    HO = orthNNLS(X, ZS, Xn)
    # Transposition
    Z = HO.T
    w, v = extract_w_v(Z)
    return w, v, Z


def initialize_w_values(Xl, v):
    n = len(v)
    w = np.zeros(n)
    # compute the degrees of the nodes
    degrees = np.array(Xl.sum(axis=1)).ravel()
    # compute the sum of node degrees for each community
    d_r = np.bincount(v, weights=degrees)

    w = np.divide(
        degrees,
        d_r[v],
        out=np.zeros_like(degrees, dtype=float),
        where=d_r[v] != 0
    )
    return w

def extract_w_v(Z):
    """ Extracts w and v from Z."""
    w = np.max(Z, axis=1)
    r = Z.shape[1]
    v = np.argmax(Z, axis=1)

    
    rows_all_zero = np.all(Z == 0, axis=1)
    v[rows_all_zero] = np.random.randint(0, r, size=np.sum(rows_all_zero))
    return w, v
def orthNNLS(M, U, Mn=None):
    """
    Solves the following optimization problem:
    min_{norm2v >= 0, V >= 0 and VV^T = D} ||M - U * V||_F^2

    Parameters:
        M (numpy.ndarray or csr_matrix): Matrix M of size (m, n).
        U (numpy.ndarray ): Matrix U of size (m, r).
        Mn (numpy.ndarray or csr_matrix, optional): Normalized columns of M. If None, it will be computed.

    Returns:
        V (numpy.ndarray): The matrix V of size (r, n) that approximates M.
        norm2v (numpy.ndarray): The squared norms of the columns of V.
    see F. Pompili, N. Gillis, P.-A. Absil and F. Glineur, "Two Algorithms for
    Orthogonal Nonnegative Matrix Factorization with Application to
    Clustering", Neurocomputing 141, pp. 15-25, 2014.
    """

    if Mn is None:
        # Normalize columns of M
        if issparse(M):
            norm2x_squared = M.multiply(M).sum(axis=0)  # matrice 1 x n (sparse)
            norm2x_squared = np.array(norm2x_squared).ravel()
            norm2x = np.sqrt(norm2x_squared)
            norm2x_safe = norm2x + 1e-16

            # Créer matrice diagonale inverses des normes
            inv_norms = 1.0 / norm2x_safe
            D = diags(inv_norms)  # matrice diagonale sparse (n x n)

            # Normaliser X par colonnes : multiplication à droite
            Mn = M @ D
        else:
            norm2m = np.sqrt(np.sum(M ** 2, axis=0))  # norm2m is the L2 norm of each column of M
            Mn = M * (1 / (norm2m + 1e-16))  # Avoid division by zero

    m, n = Mn.shape
    m_, r = U.shape

    # Normalize columns of U
    norm2u = np.sqrt(np.sum(U ** 2, axis=0))  # norm2u is the L2 norm of each column of U
    Un = U * (1 / (norm2u + 1e-16))  # Avoid division by zero
    if issparse(M):
        Mn = Mn.tocsc()
        M = M.tocsc()

    # Calculate the matrix A, which is the angles between columns of M and U
    A = Mn.T @ Un  # A is (n, r), matrix of angles

    # Find the index of the maximum value in each row of A (best column of U to approximate each column of M)
    b = np.argmax(A, axis=1)  # Indices of the best matching column in U

    # Initialize V with zeros
    V = np.zeros((r, n))

    # Assign the optimal weights to V(b(i), i)
    for i in range(n):
        if issparse(M):
            V[b[i], i] = (M[:, i].T @ U[:, b[i]] )[0] / norm2u[b[i]] ** 2
        else:
            V[b[i], i] = np.dot(M[:, i].T, U[:, b[i]]) / norm2u[b[i]] ** 2

    return V


# ------------------------------------------------
# -------------- UTILS ------------------
# ------------------------------------------------
def compute_error(normX, S):
    """ Computes error ||X - ZSZ'||_F """
    error = np.sqrt(1e-9 + normX ** 2 - np.linalg.norm(S, 'fro') ** 2) 
    return error

def compute_diff_Z(w,v,prec_w,prec_v):
    dw = w - prec_w

    same_community = (v == prec_v)

    diff_norm_sq = np.sum(
        np.where(
            same_community[None, :],
            dw**2,
            w**2 + prec_w**2
        )
    )

    previous_norm_sq = np.sum(prec_w**2)

    return np.sqrt(diff_norm_sq / previous_norm_sq)
