#'
#' Function to perform community detection on multilayer network data using DC-MASE
#' 
#' @param Adj_list a list of adjacency matrices with the same size n x n.
#' @param K number of communities. If NA, this is chosen via the scree plot.
#' @param scaled whether to use scaled or unscaled ASE.
#' @param clustering_method method to be used for clustering the vertices in the
#' embedding matrix. Options: "kmeans" or "gmm" for Gaussian mixture model.
#' @param d_vec vector of individual embedding dimensions for each graph. If NA, this is the same as K.
#' @param row.normalization parameter that indicates the option to normalize the rows of each embedding. Options: "2" for spherical spectral clustering, "score" for SCORE method by Jin (2015)
#' @param diag.augment logical. When true, the diagonal augmentation method for ASE is performed.
#' @param elbow_graph number of elbow selected in Zhu & Ghodsi method for the scree plot of each individual graph eigenvalues.
#' @param elbow_dcmase number of elbow selected in Zhu & Ghodsi method for the scree plot of the singular values of the concatenated spectral embeddings of MASE.
#' @param par whether to run in parallel (TRUE/FALSE)
#' @param numpar number of clusters for parallel implementation
#'
#' @return A list containing the estimated community memberships and the parameters of the multilayer degree-corrected stochastic block model.
#'
#' @references
#'
#' @author Jes\'us Arroyo <jarroyo@tamu.edu>
#' @export
comdet_dcmase <- function(Adj_list, K, clustering_method = "kmeans",
                          d_vec = NA, scaled = TRUE,
                          row.normalization = "2", diag.augment = FALSE,
                   elbow_graph = 1, elbow_dcmase = 2,
                   par = FALSE, numpar = 12) {
  # Obtain embedding -----------------------------------------------------------
  embed.dcmase <- dcmase(Adj_list, d = K, d_vec, scaled, row.normalization, diag.augment,
                         elbow_graph, elbow_dcmase, show.scree.results  = FALSE,
                         par, numpar)
  n <- ncol(Adj_list[[1]]) # number of vertices
  m <- length(Adj_list) # number of graphs
  
  if(is.na(K)) {
    K <- ncol(embed.dcmase) # number of communities
  }
  
  # perform clustering ---------------------------------------------------------
  if(!(clustering_method %in% c("kmeans", "gmm", "kmedians"))) {
    stop("Error in clustering method.")
  }
  if(clustering_method == "kmeans") {
    clustering <- kmeans(embed.dcmase, centers = K, nstart = 100)
    community_memberships <- clustering$cluster
  }
  if(clustering_method == "gmm") {
    require(mclust)
    clustering <- Mclust(embed.dcmase, G = K, verbose = FALSE)
    community_memberships <- clustering$classification
  }
  if(clustering_method == "kmedians") {
    require(Gmedian)
    clustering <- kGmedian(embed.dcmase, ncenters = K, nstart = 100)
    community_memberships <- clustering$cluster
  }
  Z <- matrix(0, n, K)
  Z[cbind(1:n, community_memberships)] <- 1
  
  # Estimate model parameters given the clusters -------------------------------
  degree_corrections <- matrix(0, n, m)
  B_matrices <- array(0, c(K, K, m))
  
  # For degree correct. use d_i/sum(d_i per community) * n_k
  n_k <- colSums(Z)
  degrees <- lapply(Adj_list, rowSums)
  normalizations <- lapply(degrees, function(degs) crossprod(Z, degs)/n_k)
  degree_corrections <- sapply(1:m, function(i) degrees[[i]]/(Z %*% normalizations[[i]]))
  
  # For connectivity, calculate average edges per cell
  nk_nk <- tcrossprod(n_k)
  B_matrices <- plyr::laply(Adj_list, function(A) 
    matrix(crossprod(Z, crossprod(A, Z)) / nk_nk, K, K) )
  B_matrices <- array(B_matrices, dim = c(m, K, K))
  # Order communities according to total degree
  Abar <- Reduce("+", Adj_list)
  degrees_all <- colSums(Abar)
  comsizes <- crossprod(Z, degrees_all)
  ordercoms <- rank(-comsizes)
  return(list(community_memberships = ordercoms[community_memberships], 
              Z = Z[, ordercoms, drop = F], 
              degree_corrections = degree_corrections, 
              B_matrices = B_matrices[, ordercoms, ordercoms, drop = F], 
              V = embed.dcmase))
}




calculate_dcsbm_parameters <- function(Adj_list, community_memberships) {
  
  n <- ncol(Adj_list[[1]])
  m <- length(Adj_list)
  K <- length(unique(community_memberships))
  
  Z <- matrix(0, n, K)
  Z[cbind(1:n, community_memberships)] <- 1
  
  degree_corrections <- matrix(0, n, m)
  B_matrices <- array(0, c(K, K, m))
  degree_corrections_normalized <- matrix(0, n, m)
  B_matrices_normalized <- array(0, c(K, K, m))
  
  # For degree correct. use d_i/sum(d_i per community) * n_k
  n_k <- colSums(Z)
  degrees <- lapply(Adj_list, rowSums)
  normalizations <- lapply(degrees, function(degs) crossprod(Z, degs)/n_k)
  degree_corrections <- sapply(1:m, function(i) degrees[[i]]/(Z %*% normalizations[[i]]))
  
  # For connectivity, calculate average edges per cell
  nk_nk <- tcrossprod(n_k)
  B_matrices <- plyr::laply(Adj_list, function(A) 
    matrix(crossprod(Z, crossprod(A, Z)) / nk_nk, K, K) )
  B_matrices <- array(B_matrices, dim = c(m, K, K))
  # Order communities according to total degree
  Abar <- Reduce("+", Adj_list)
  degrees_all <- colSums(Abar)
  comsizes <- crossprod(Z, degrees_all)
  ordercoms <- rank(-comsizes)
  
  
  degree_corrections_normalized <- sapply(1:m, function(i) degree_corrections[,i] * ((Z %*% diag(sqrt(diag(B_matrices[i, ,])))) %*% rep(1,K)))
  for(i in 1:m) {
    B_matrices_normalized[i, , ] <- diag(1/sqrt(diag(B_matrices[i, ,])))  %*% B_matrices[i, ,] %*% diag(1/sqrt(diag(B_matrices[i, ,])))
  }
  return(list(degree_corrections = degree_corrections,
              B_matrices = B_matrices,
              degree_corrections_normalized = degree_corrections_normalized,
              B_matrices_normalized = B_matrices_normalized))
}


