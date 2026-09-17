#'
#' Function to perform community detection on multilayer network data
#' @references
#'
#' @author Jes\'us Arroyo <jarroyo@tamu.edu>
#' @export
#' 
#' 

library(reticulate)
use_python("/home/pistou/miniconda3/envs/mdcbm/bin/python",
           required = TRUE)
sys <- import("sys")
sys_path <- path.expand("~/MDCBM/Python")
sys$path <- c(sys$path, sys_path)
frost <- import("frost.frost_multilayer")
frost_sharedZ <- import("frost.frost_multilayer_sharedZ")
np <- import("numpy")
source("R/Codes_Spectral_Matrix_Paul_Chen_AOS_2020.r")
source("R/comdet-dcmase.R")
source("R/dcmase.R")
source("R/SpectralMethods.R")
source("R/run_graph_tool.R")

# Regularized normalization function for one adjacency matrix
normalize_adj <- function(A, tau = NULL, tau_frac = 1) {
  A <- as.matrix(A)
  d <- rowSums(A)                 # degrees (assumes A is symmetric)
  
  if (is.null(tau)) {
    tau <- tau_frac * mean(d)     # average-degree rule: tau = mean(d)
  }
  
  d_reg <- d + tau
  D_inv_sqrt <- 1 / sqrt(d_reg)   # vector, not the full matrix
  
  # A_tilde[i,j] = A[i,j] / sqrt(d_reg[i] * d_reg[j])
  A_tilde <- sweep(A, 1, D_inv_sqrt, "*")
  A_tilde <- sweep(A_tilde, 2, D_inv_sqrt, "*")
  
  A_tilde
}


comdetmethods <- function(Adj_list, K, method) {
  n <- ncol(Adj_list[[1]])
  
  #################################################################################
  if(method == "dcmase") {
    community_memberships <- comdet_dcmase(Adj_list, K, "kmeans")$community_memberships
  }
  #################################################################################
  if(method == "dcmase-gmm") {
    community_memberships <- comdet_dcmase(Adj_list, K, "gmm")$community_memberships
  }
  #################################################################################
  if(method == "dcmase-unscaled") {
    community_memberships <- comdet_dcmase(Adj_list, K,scaled = FALSE, "kmeans")$community_memberships
  }
  #################################################################################
  if(method == "mase") {
    mase.res <- mase(Adj_list, d = K, scaled.ASE = FALSE, diag.augment = FALSE)
    community_memberships <- kmeans(mase.res$V, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "mase-gmm") {
    require(mclust)
    mase.res <- mase(Adj_list, d = K, scaled.ASE = FALSE, diag.augment = FALSE)
    community_memberships <- Mclust(mase.res$V, K, verbose = FALSE)$classification
  }
  #################################################################################
  if(method == "mase-spherical") {
    require(mclust)
    mase.res <- mase(Adj_list, d = K, scaled.ASE = FALSE, diag.augment = FALSE)
    V <- mase.res$V
    rownorms <- apply(V, 1,function(x) sqrt(sum(x^2)))
    rownorms <- rownorms + 1*(abs(rownorms) < 1e-15)
    community_memberships <- kmeans(V/rownorms, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "mase-score") {
    community_memberships <- comdet_dcmase(Adj_list, K,row.normalization = "score", "kmeans")$community_memberships
  }
  #################################################################################
  if(method == "ave") {
    Abar <- Reduce("+", Adj_list)
    V <- eig_embedding(Abar, K)
    community_memberships <- kmeans(V, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "ave_spherical") {
    Abar <- Reduce("+", Adj_list)
    V <- eig_embedding(Abar, K)
    rownorms <- apply(V, 1,function(x) sqrt(sum(x^2)))
    rownorms <- rownorms + 1*(abs(rownorms) < 1e-15)
    
    community_memberships <- kmeans(V / rownorms, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "score") {
    Abar <- Reduce("+", Adj_list)
    V <- eig_embedding(Abar, K)
    V <- V[, 2:K, drop = F] / (V[,1] + 1*(abs(V[,1]) < 1e-15))
    community_memberships <- kmeans(V, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "sq-bias-adjusted") {
    # https://arxiv.org/pdf/2003.08222.pdf
    Asq_bias_removed <- lapply(Adj_list, function(A) {
      d = colSums(A) # degrees
      A2 <- crossprod(A)
      diag(A2) <- diag(A2) - d
      A2
    })
    Asq_bias_sum <- Reduce("+", Asq_bias_removed)
    V <- eig_embedding(Asq_bias_sum, K)
    rownorms <- apply(V, 1,function(x) sqrt(sum(x^2)))
    rownorms <- rownorms + 1*(abs(rownorms) < 1e-15)
    community_memberships <- kmeans(V/rownorms, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "omni") {
    if(length(Adj_list)==1) {
      omnibar = g.ase(Adj_list[[1]], K, diag.augment = F)$X
    } else{
      omni <- OMNI_matrix(Adj_list, K)
      omnibar <- Reduce("+", lapply(omni, function(x) x$X)) 
    }
    community_memberships <- kmeans(omnibar, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "lmfo") {
    # https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-1/Spectral-and-matrix-factorization-methods-for-consistent-community-detection-in/10.1214/18-AOS1800.short
    community_memberships <- lmfo(lapply(Adj_list, as.matrix), n, K)
  }
  #################################################################################
  if(method == "speck") {
    # # https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-1/Spectral-and-matrix-factorization-methods-for-consistent-community-detection-in/10.1214/18-AOS1800.short
    community_memberships <- speck(Adj_list, n, K)
  }
  #################################################################################
  if(method == "graph-tool") {
    # Note: this method requires graph-tool library to be installed https://graph-tool.skewed.de/
    community_memberships <- run_graph_tool(Adj_list, K)
  }
  #################################################################################
  if(method == "score-onenetwork") {
    m <-length(Adj_list)
    V <- eig_embedding(Adj_list[[sample(m,1)]], K)
    V <- V[, 2:K, drop = F] / (V[,1] + 1*(abs(V[,1]) < 1e-15))
    community_memberships <- kmeans(V, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "spherical-onenetwork") {
    m <-length(Adj_list)
    V <- eig_embedding(Adj_list[[sample(m,1)]], K)
    rownorms <- apply(V, 1,function(x) sqrt(sum(x^2)))
    rownorms <- rownorms + 1*(abs(rownorms) < 1e-15)
    community_memberships <- kmeans(V/rownorms, K, nstart = 100)$cluster
  }
   #################################################################################
    

    if (method=="mfrost"){
      seed_for_python <- sample.int(.Machine$integer.max, 1)
      
      Adj_list_numpy <- lapply(Adj_list, function(X) np$array(as.matrix(X)))
      X_list <- r_to_py(Adj_list_numpy)
      
      res <- frost$frost_multilayer(X_list, K, init_method='USENC',init_seed=seed_for_python,numTrials=3L,time_limit=1000)
      labels <- py_to_r(res[[2]])
      community_memberships <- as.vector(labels + 1)
    }
   # Initialized by DC_MASE
    if (method=="frost-dcmase"){
     
      community_memberships <- comdet_dcmase(Adj_list, K, "kmeans")$community_memberships
      init_partition=np$array(community_memberships-1,dtype="int32")
      seed_for_python <- sample.int(.Machine$integer.max, 1)
      
      Adj_list_numpy <- lapply(Adj_list, function(X) np$array(as.matrix(X)))
      X_list <- r_to_py(Adj_list_numpy)
      
      res <- frost$frost_multilayer(X_list, K, init_partition=init_partition,init_seed=seed_for_python,numTrials=3L,time_limit=1000)
      labels <- py_to_r(res[[2]])
      community_memberships <- as.vector(labels + 1)
    }
    
      if (method=="frost-sharedZ-Anorm"){
      
      seed_for_python <- sample.int(.Machine$integer.max, 1)
      
      # Apply to each layer, with a layer-specific tau (recommended in the multilayer setting)
      Adj_list_norm <- lapply(Adj_list, function(A) normalize_adj(A))

      # Then convert to numpy as before
      Adj_list_numpy <- lapply(Adj_list_norm, function(X) np$array(as.matrix(X)))
      X_list <- r_to_py(Adj_list_numpy)
    
      res <- frost_sharedZ$frost_multilayer_sharedZ(X_list, K, init_method='MF-SC-CA',init_seed=seed_for_python,numTrials=3L,time_limit=1000)
      labels <- py_to_r(res[[2]])
      community_memberships <- as.vector(labels + 1)
       
    }
    if (method=="frost-sharedZ"){
      
      seed_for_python <- sample.int(.Machine$integer.max, 1)
      
      Adj_list_numpy <- lapply(Adj_list, function(X) np$array(as.matrix(X)))
      X_list <- r_to_py(Adj_list_numpy)
    
      res <- frost_sharedZ$frost_multilayer_sharedZ(X_list, K, init_method='MF-SC-CA',init_seed=seed_for_python,numTrials=3L,time_limit=1000)
      labels <- py_to_r(res[[2]])
      community_memberships <- as.vector(labels + 1)
       
    }

  return(community_memberships)
}

allmethods <- function(Adj_list, K, method) {
  n <- ncol(Adj_list[[1]])
  
  #################################################################################
  if(method == "dcmase") {
    community_memberships <- comdet_dcmase(Adj_list, K, "kmeans")$community_memberships
  }
  #################################################################################
  if(method == "dcmase-gmm") {
    community_memberships <- comdet_dcmase(Adj_list, K, "gmm")$community_memberships
  }
  #################################################################################
  if(method == "dcmase-unscaled") {
    community_memberships <- comdet_dcmase(Adj_list, K,scaled = FALSE, "kmeans")$community_memberships
  }
  #################################################################################
  if(method == "mase") {
    mase.res <- mase(Adj_list, d = K, scaled.ASE = FALSE, diag.augment = FALSE)
    community_memberships <- kmeans(mase.res$V, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "mase-gmm") {
    require(mclust)
    mase.res <- mase(Adj_list, d = K, scaled.ASE = FALSE, diag.augment = FALSE)
    community_memberships <- Mclust(mase.res$V, K, verbose = FALSE)$classification
  }
  #################################################################################
  if(method == "mase-spherical") {
    require(mclust)
    mase.res <- mase(Adj_list, d = K, scaled.ASE = FALSE, diag.augment = FALSE)
    V <- mase.res$V
    rownorms <- apply(V, 1,function(x) sqrt(sum(x^2)))
    rownorms <- rownorms + 1*(abs(rownorms) < 1e-15)
    community_memberships <- kmeans(V/rownorms, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "mase-score") {
    community_memberships <- comdet_dcmase(Adj_list, K,row.normalization = "score", "kmeans")$community_memberships
  }
  #################################################################################
  if(method == "ave") {
    Abar <- Reduce("+", Adj_list)
    V <- eig_embedding(Abar, K)
    community_memberships <- kmeans(V, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "ave_spherical") {
    Abar <- Reduce("+", Adj_list)
    V <- eig_embedding(Abar, K)
    rownorms <- apply(V, 1,function(x) sqrt(sum(x^2)))
    rownorms <- rownorms + 1*(abs(rownorms) < 1e-15)
    
    community_memberships <- kmeans(V / rownorms, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "score") {
    Abar <- Reduce("+", Adj_list)
    V <- eig_embedding(Abar, K)
    V <- V[, 2:K, drop = F] / (V[,1] + 1*(abs(V[,1]) < 1e-15))
    community_memberships <- kmeans(V, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "sq-bias-adjusted") {
    # https://arxiv.org/pdf/2003.08222.pdf
    Asq_bias_removed <- lapply(Adj_list, function(A) {
      d = colSums(A) # degrees
      A2 <- crossprod(A)
      diag(A2) <- diag(A2) - d
      A2
    })
    Asq_bias_sum <- Reduce("+", Asq_bias_removed)
    V <- eig_embedding(Asq_bias_sum, K)
    rownorms <- apply(V, 1,function(x) sqrt(sum(x^2)))
    rownorms <- rownorms + 1*(abs(rownorms) < 1e-15)
    community_memberships <- kmeans(V/rownorms, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "omni") {
    if(length(Adj_list)==1) {
      omnibar = g.ase(Adj_list[[1]], K, diag.augment = F)$X
    } else{
      omni <- OMNI_matrix(Adj_list, K)
      omnibar <- Reduce("+", lapply(omni, function(x) x$X)) 
    }
    community_memberships <- kmeans(omnibar, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "lmfo") {
    # https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-1/Spectral-and-matrix-factorization-methods-for-consistent-community-detection-in/10.1214/18-AOS1800.short
    community_memberships <- lmfo(lapply(Adj_list, as.matrix), n, K)
  }
  #################################################################################
  if(method == "speck") {
    # # https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-1/Spectral-and-matrix-factorization-methods-for-consistent-community-detection-in/10.1214/18-AOS1800.short
    community_memberships <- speck(Adj_list, n, K)
  }
  #################################################################################
  if(method == "graph-tool") {
    # Note: this method requires graph-tool library to be installed https://graph-tool.skewed.de/
    community_memberships <- run_graph_tool(Adj_list, K)
  }
  #################################################################################
  if(method == "score-onenetwork") {
    m <-length(Adj_list)
    V <- eig_embedding(Adj_list[[sample(m,1)]], K)
    V <- V[, 2:K, drop = F] / (V[,1] + 1*(abs(V[,1]) < 1e-15))
    community_memberships <- kmeans(V, K, nstart = 100)$cluster
  }
  #################################################################################
  if(method == "spherical-onenetwork") {
    m <-length(Adj_list)
    V <- eig_embedding(Adj_list[[sample(m,1)]], K)
    rownorms <- apply(V, 1,function(x) sqrt(sum(x^2)))
    rownorms <- rownorms + 1*(abs(rownorms) < 1e-15)
    community_memberships <- kmeans(V/rownorms, K, nstart = 100)$cluster
  }
   #################################################################################
    if (method=="frost-mf"){
      
      seed_for_python <- sample.int(.Machine$integer.max, 1)
      
      Adj_list_numpy <- lapply(Adj_list, function(X) np$array(as.matrix(X)))
      X_list <- r_to_py(Adj_list_numpy)
    
      res <- frost$frost_multilayer(X_list, K, init_method='MF-SC-CA',init_seed=seed_for_python,numTrials=50L,time_limit=1000)
      labels <- py_to_r(res[[2]])
      community_memberships <- labels + 1
       
    }

    if (method=="frost-us"){
      seed_for_python <- sample.int(.Machine$integer.max, 1)
      
      Adj_list_numpy <- lapply(Adj_list, function(X) np$array(as.matrix(X)))
      X_list <- r_to_py(Adj_list_numpy)
      
      res <- frost$frost_multilayer(X_list, K, init_method='USENC',init_seed=seed_for_python,numTrials=10L,time_limit=1000)
      labels <- py_to_r(res[[2]])
      community_memberships <- labels + 1
    }

    if (method=="frost-dcmase"){
     
      community_memberships <- comdet_dcmase(Adj_list, K, "kmeans")$community_memberships
      init_partition=np$array(community_memberships-1,dtype="int32")
      seed_for_python <- sample.int(.Machine$integer.max, 1)
      
      Adj_list_numpy <- lapply(Adj_list, function(X) np$array(as.matrix(X)))
      X_list <- r_to_py(Adj_list_numpy)
      
      res <- frost$frost_multilayer(X_list, K, init_partition=init_partition,init_seed=seed_for_python,numTrials=50L,time_limit=1000)
      labels <- py_to_r(res[[2]])
      community_memberships <- labels + 1
    }
    if (method=="mf"){
      
      
      seed_for_python <- sample.int(.Machine$integer.max, 1)
      
      Adj_list_numpy <- lapply(Adj_list, function(X) np$array(as.matrix(X)))
      X_list <- r_to_py(Adj_list_numpy)
     
      res <- frost$frost_multilayer(X_list, K, maxiter=as.integer(0), init_method='MF-SC-CA',init_seed=seed_for_python)
      labels <- py_to_r(res[[2]])
      community_memberships <- labels + 1
      }
      if (method=="us"){
      
      
      seed_for_python <- sample.int(.Machine$integer.max, 1)
      
      Adj_list_numpy <- lapply(Adj_list, function(X) np$array(as.matrix(X)))
      X_list <- r_to_py(Adj_list_numpy)

          
      res <- frost$frost_multilayer(X_list, K, maxiter=as.integer(0),init_method='USENC',init_seed=seed_for_python)
      labels <- py_to_r(res[[2]])
      community_memberships <- labels + 1 
      }

  return(community_memberships)
}



