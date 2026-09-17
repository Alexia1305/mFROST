import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix

'''
Methods to find a consensus between several node partitions (including the used USENC method)
Used in our initialization procedure in mFROST

'''
def clustering_coassociation(W):
    L, n, r = W.shape
    C = np.zeros((n, n))

    for Wl in W:
        for c_idx in range(r):
            members = np.where(Wl[:, c_idx] > 0)[0]
            C[np.ix_(members, members)] += 1

    C /= L
    return C

def coassocation_matrix(v,r):
    L, n = v.shape
    C = np.zeros((n, n))

    for layer in range(L):
        for c_idx in range(r):
            members = np.where(v[layer, :] == c_idx)[0]
            C[np.ix_(members, members)] += 1

    C /= L
    return C


def normalized_laplacian(C):
    """
    Compute the symmetric normalized Laplacian L_sym = I - D^{-1/2} C D^{-1/2}

    Parameters
    ----------
    C : ndarray of shape (n, n)
        Symmetric similarity (co-association) matrix.

    Returns
    -------
    L_sym : ndarray of shape (n, n)
        Symmetric normalized Laplacian.
    """
    # Degrés
    d = C.sum(axis=1)

    # Inverse racine des degrés
    d_inv_sqrt = 1.0 / np.sqrt(d)

    # Protection contre les degrés nuls
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    # Normalisation
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(C.shape[0]) - D_inv_sqrt @ C @ D_inv_sqrt

    return L_sym




class MatrixFreeSpectralClusteringCoAssociation:
    """
    Spectral clustering using the normalized Laplacian of the co-association matrix C
    without explicity computing C
    """

    def __init__(self, partitions, n_clusters):
        """
            Pre computations O(Ln) for the matrix-vector product
            with the normalized graph Laplacian associated with the co-association matrix of the L partitions v.

            Parameters
            ----------
            partitions : ndarray of shape (L, n)
                Matrix of partitions
            n_clusters : int
                Number of communities
        """
        self.partitions = partitions.astype(int)
        self.n_clusters = n_clusters
        self.L, self.n = partitions.shape
        # Compute the permutation vector of the nodes to order the partition for each layer
        # given the cluster indexes
        # and compute the size of each cluster for each layer
        self.permutations = np.zeros((self.L, self.n), dtype=np.int64)
        self.max_clusters = np.max(self.partitions)+1
        self.size_clusters = np.zeros((self.L, self.max_clusters), dtype=np.int64)

        for layer in range(self.L): #O(Ln)
            self.permutations[layer], self.size_clusters[layer] = self.counting_sort_with_perm(self.partitions[layer])

        # Computing d_norm= 1/sqrt(d) d(i,i)=sum_j C(i,j)
        layers = np.arange(self.L)[:, None]  # shape (L, 1)
        # degree of the node i = sum of the size of each cluster
        d = self.size_clusters[layers, self.partitions]  # shape (L, n)
        d = d.sum(axis=0)  # somme sur les L layers
        self.d_norm = 1.0 / np.sqrt(d)

    def counting_sort_with_perm(self, partition):
        """
        Compute the size of each clusters in partition and a permutation vector
        to sort the nodes according to their cluster index

        Parameters
        ----------
        partition : ndarray of shape (n,)
            Cluster index for the n nodes

        Returns
        -------
        permutation : ndarray of shape (n,)
            permutation vector to sort the nodes
        cluster_sizes : ndarray of shape (self.n_clusters,)
            Number of nodes in each clusters


        """
        n = len(partition)

        # Compute the size of each cluster
        cluster_size = np.bincount(partition, minlength=self.max_clusters)  # O(n)

        # Compute the last position of each cluster
        end_cluster = cluster_size.copy()
        for i in range(1, self.max_clusters):  #O(n)
            end_cluster[i] += end_cluster[i - 1]

        # Compute the permutation
        permutation = np.zeros(n, dtype=np.int64)

        for i in range(n - 1, -1, -1): # O(n)
            x = partition[i]
            end_cluster[x] -= 1
            pos = end_cluster[x]
            permutation[pos] = i

        return permutation, cluster_size
    def matvec(self,x):
        """
        Compute the matrix-vector product with the normalized graph Laplacian
        associated with the co-association matrix. O(Ln)

        The normalized Laplacian is defined as:
            L_sym = I - D^{-1/2} C D^{-1/2}

        This function computes:
            y = L_sym @ x

        Parameters
        ----------
        x : ndarray of shape (n,)
            Input vector

        Returns
        -------
        y : ndarray of shape (n,)
            Result of the matrix-vector product L_sym @ x.

        Notes
        -----
        - This implementation avoids explicitly forming the co-association matrix
        """
        # Compute dx= D^(-1/2)x
        dx = self.d_norm*x # O(n)

        # compute Cdx=C*D^(-1/2)x
        cdx = np.zeros(self.n)
        for layer in range(self.L): #O(Ln)
            start = 0
            for cluster_idx in range(self.max_clusters):
                # nodes in cluster r
                nodes_in_this_cluster = np.arange(start, start+self.size_clusters[layer, cluster_idx])
                cdx[self.permutations[layer,  nodes_in_this_cluster]] += np.sum(dx[self.permutations[layer,  nodes_in_this_cluster]])

                start += self.size_clusters[layer, cluster_idx]

        #Compute Lsym@x=x-D^(-1/2)C*D^(-1/2)x

        return x-self.d_norm*cdx #O(n)

    def fit_predict(self):
        """
        Perform spectral clustering using only the matvec (L_sym @ x).

        Returns
        -------
        labels : ndarray of shape (n_nodes,)
            Cluster assignment for each node.

        Notes
        -----
        - This implementation avoids explicitly forming the co-association matrix
        """

        # Créer un opérateur linéaire "LinearOperator"
        M_op = LinearOperator(shape=(self.n, self.n), matvec=self.matvec)

        # Calculer les k plus petites valeurs propres
        # which='SA' -> Smallest Algebraic
        vals, vecs = eigsh(M_op, k=self.n_clusters, which='SA') #O(tnL)
        # U = matrice des k vecteurs propres
        U = vecs

        # Normalisation des lignes
        U_norm = U / np.linalg.norm(U, axis=1, keepdims=True) #O(nr)

        # Clustering
        kmeans = KMeans(n_clusters=self.n_clusters) #O(tnr^2)
        return kmeans.fit_predict(U_norm)

# -----------------------------
# USENC Consensus Function
# -----------------------------
def USENC_ConsensusFunction(baseCls, k):
    """
    Combine the M node partitions in baseCls to obtain the final clustering (concensus node partition) result (with k clusters).
     Based on the paper "Ultra-scalable spectral clustering and ensemble clustering", IEEE TKDE, 2019.
    """
    baseCls=baseCls.astype(int)
    N, M = baseCls.shape

    # Compute number of clusters per partition
    nCls_per_part = np.max(baseCls, axis=0) + 1  # because labels start at 0

    # Each cluster for the M partition has its own number 
    offsets = np.zeros(M, dtype=np.int64)
    offsets[1:] = np.cumsum(nCls_per_part[:-1])

    # Apply offsets 
    baseCls_offset = baseCls + offsets  # broadcast

    # total number of clusters
    cntCls = int(np.sum(nCls_per_part))

    # Build bipartite graph nodes/ clusters partition
    rows = np.repeat(np.arange(N), M).astype(np.int64)
    cols = baseCls_offset.flatten().astype(np.int64)
    data = np.ones(N * M)
    B = coo_matrix((data, (rows, cols)), shape=(N, cntCls)).tocsc()

    # Remove empty columns
    col_sum = B.sum(axis=0).A1  # convert to 1D array
    B = B[:, col_sum > 0]

    # Cut the bipartite graph to obtain the consensus partition 
    labels = Tcut_for_bipartite_graph(B, k)
    return labels

# -----------------------------
# Tcut for Bipartite Graph
# -----------------------------
def Tcut_for_bipartite_graph(B, Nseg, maxKmIters=100, cntReps=3):
    """
    B - |X|-by-|Y| cross-affinity matrix
    """
    Nx, Ny = B.shape
    if Ny < Nseg:
        raise ValueError('Need more columns!')

    # Compute Dx = diag(1 / sum of rows)
    dx = np.array(B.sum(axis=1)).flatten()
    dx[dx == 0] = 1e-10  # avoid division by zero
    Dx = coo_matrix((1.0 / dx, (np.arange(Nx), np.arange(Nx))), shape=(Nx, Nx)).tocsc()

    # Compute Wy = B^T * Dx * B
    Wy = (B.T @ Dx @ B).tocsc()

    # Compute normalized affinity matrix
    d = np.array(Wy.sum(axis=1)).flatten()
    D = coo_matrix((1.0 / np.sqrt(d), (np.arange(Ny), np.arange(Ny))), shape=(Ny, Ny)).tocsc()
    nWy = D @ Wy @ D
    nWy = (nWy + nWy.T) / 2  # make symmetric

    # Compute eigenvectors
    evals, evecs = eigh(nWy.toarray())
    idx = np.argsort(evals)[::-1]
    Ncut_evec = D @ evecs[:, idx[:Nseg]]

    # Transfer Ncut eigenvectors to entire bipartite graph
    evec = Dx @ B @ Ncut_evec

    # Normalize each row to unit norm
    norms = np.linalg.norm(evec, axis=1, keepdims=True) + 1e-10
    evec = evec / norms

    # k-means clustering
    kmeans = KMeans(n_clusters=Nseg, max_iter=maxKmIters, n_init=cntReps, random_state=0)
    labels = kmeans.fit_predict(evec)
    return labels
























