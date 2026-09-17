library(Matrix)
library(igraph)
library(mclust)
library(aricode)
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")
source("R/Codes_Spectral_Matrix_Paul_Chen_AOS_2020.r")
source("R/comdet-dcmase.R")
source("R/comdetmethods.R")
source("R/dcmase.R")
source("R/SpectralMethods.R")
source("R/run_graph_tool.R")
source("R/make_ggplot.R")



run_simulations <- function(sim_setting, parameters, repetitions = 20) {
  library(parallel)
  cl = makeCluster(10)
clusterEvalQ(cl, {
  Sys.setenv(OMP_NUM_THREADS = "1")
  Sys.setenv(MKL_NUM_THREADS = "1")
})
  clusterEvalQ(cl = cl, source("Experiments/run_all_methods.R"))
  clusterEvalQ(cl = cl, source("Experiments/extrasimulations.R"))
  clusterExport(cl = cl, varlist = c("sim_setting", "parameters"),envir = environment()) 
  results <- parLapply(cl, 1:repetitions, function(seed) {
    generate_data <- sim_setting(parameters, seed)
    res <- run_all_methods(generate_data$Adj_list,
                         generate_data$truecom,seed)

    res <- data.frame(
      Seed = seed,
      Method = rownames(res),
      res,
      row.names = NULL
    )

    res
})
 df_res <- do.call(rbind, results)
  stopCluster(cl)
  return(df_res)
}

iterate_parameters <- function(sim_setting, parameters_list, param_iter, 
                               repetitions = 20) {
  df_res <- lapply(1:length(parameters_list),  function(i) {
    cat("Running parameter ", param_iter[[i]], "...\n", sep = "")
    sim_res <- run_simulations(sim_setting, parameters = parameters_list[[i]], repetitions)
    sim_res$parameter <- param_iter[[i]]
    return(sim_res)
  })
  return(Reduce(rbind, df_res))
}

# run_simulations <- function(sim_setting, parameters, repetitions = 20) {
  
#   results <- lapply(1:repetitions, function(seed) {
#     cat("Seed:", seed, "\n")
#     generate_data <- sim_setting(parameters, seed)
#     res <- run_all_methods(generate_data$Adj_list,
#                          generate_data$truecom)

#     res <- data.frame(
#       Seed = seed,
#       Method = rownames(res),
#       res,
#       row.names = NULL
#     )

#     res
# })
#  df_res <- do.call(rbind, results)
  
#   return(df_res)
# }

# iterate_parameters <- function(sim_setting, parameters_list, param_iter, 
#                                repetitions = 20) {
#   df_res <- lapply(1:length(parameters_list),  function(i) {
#     cat("Running parameter ", param_iter[[i]], "...\n", sep = "")
#     sim_res <- run_simulations(sim_setting, parameters = parameters_list[[i]], repetitions)
#     sim_res$parameter <- param_iter[[i]]
#     return(sim_res)
#   })
#   return(Reduce(rbind, df_res))
# }



run_all_methods <- function(Adj_list, truecoms,seed) {
  K <- length(unique(truecoms))
  
  # Note: to run graph-tool, uncomment the following lines and comment the next
  # methods_to_run <- c("mfrost","dcmase", "ave_spherical", "sq-bias-adjusted","mase-spherical","lmfo", "graph-tool")
  methods_to_run <- c("mfrost","frost-sharedZ-Anorm","frost-sharedZ")
 
  results <- lapply(methods_to_run, function(method) {
    print(method)
    set.seed(seed)
  pred <- comdetmethods(Adj_list, K, method = method)

  err <- classError(pred, truecoms)$errorRate
  nmi <- NMI(pred, truecoms)
  ari <- adjustedRandIndex(pred, truecoms)

  c(
    Error = err,
    NMI = nmi,
    ARI = ari
  )
})

results <- do.call(rbind, results)
#rownames(results) <- c("mFROST",  "DC-MASE","Sum A","Bias-adjusted SoS", "MASE","OLMF","graph-tool")
rownames(results) <- c("mFROST","mFROST-Zshared-L","mFROST-Zshared")

return(results)
}



#########################################################################
### Simulation 1: same B same theta
simulation1 <- function(parameters, seed = 1989) {
  set.seed(seed)
  m <- parameters$m
  ave_deg <- 10
  alpha <- 2.5
  # Generate network parameters ------------------------------------------------
  n <- parameters$n
  K <- parameters$K
 
  Z <- kronecker(diag(K), rep(1, n/K))
  if (parameters$degree_distribution=="exp"){
    theta <- rexp(n) + 0.2
  }
  else if (parameters$degree_distribution=="pow") {
     theta <- runif(min=0, max=1,n)^(-1/(alpha-1))
  }
  theta <- as.vector(theta / (Z%*%crossprod(Z,theta)/(n/K)))
  
  B <- 0.06*diag(K) + 0.04
  
  degree_corrections <- lapply(1:m, function(i) theta)
  B_matrices <- lapply(1:m, function(i) B)
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  return(list(Adj_list = Adj_list, truecom = as.vector(Z %*% 1:K)))
}

### Simulation 2: same theta, different B
simulation2 <- function(parameters, seed = 1989) {
   set.seed(seed)
  m <- parameters$m
  ave_deg <- 10
  alpha <- 2.5
  # Generate network parameters ------------------------------------------------
  n <- parameters$n
  K <- parameters$K
  Z <- kronecker(diag(K), rep(1, n/K))
  
  if (parameters$degree_distribution=="exp"){
    theta <- rexp(n) + 0.2
  }
  else if (parameters$degree_distribution=="pow") {
     theta <- runif(min=0,max=1,n)^(-1/(alpha-1))
  }
  theta <- as.vector(theta / (Z%*%crossprod(Z,theta)/(n/K)))
  degree_corrections <- lapply(1:m, function(i) theta)
  
  B_matrices <- lapply(1:m, function(i) {
    p <- runif(1)
    q <- runif(1)
    (p-q) *diag(K) + q
  })
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  return(list(Adj_list = Adj_list, truecom = as.vector(Z %*% 1:K)))
}







### Simulation 3: different B, different theta
simulation3 <- function(parameters, seed = 1989) {
  set.seed(seed)
  m <- parameters$m
  ave_deg <- 10
  alpha <- 2.5
  # Generate network parameters ------------------------------------------------
  n <- parameters$n
  K <- parameters$K
  Z <- kronecker(diag(K), rep(1, n/K))
  
  #degree_corrections <- lapply(1:m, function(i) runif(n, min = sqrt(0.05), max = 1)^2)

  degree_corrections <- lapply(1:m, function(i) {
    if (parameters$degree_distribution=="exp"){
    theta <- rexp(n) + 0.2
  }
  else if (parameters$degree_distribution=="pow") {
     theta <- runif(min=0,max=1,n)^(-1/(alpha-1))
  }
    as.vector(theta / (Z%*%crossprod(Z,theta)/(n/K)))
  })
  
  
  
  
  B_matrices <- lapply(1:m, function(i) {
    p <- runif(1)
    q <- runif(1)
    (p-q) *diag(K) + q
  })
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  return(list(Adj_list = Adj_list, truecom = as.vector(Z %*% 1:K)))
}


### Simulation 4: same B, different theta
simulation4 <- function(parameters, seed = 1989) {
   set.seed(seed)
  m <- parameters$m
  ave_deg <- 10
  alpha <- 2.5
  # Generate network parameters ------------------------------------------------
  n <- parameters$n
  K <- parameters$K
  Z <- kronecker(diag(K), rep(1, n/K))
  
  degree_corrections <- lapply(1:m, function(i) {
    if (parameters$degree_distribution=="exp"){
    theta <- rexp(n) + 0.2
  }
  else if (parameters$degree_distribution=="pow") {
     theta <- runif(min=0,max=1,n)^(-1/(alpha-1))
  }
    as.vector(theta / (Z%*%crossprod(Z,theta)/(n/K)))
  })
  B_matrices <- lapply(1:m, function(i) {
    B <- 0.06*diag(K) + 0.04
    B
  })
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  return(list(Adj_list = Adj_list, truecom = as.vector(Z %*% 1:K)))
}

### Simulation 4b: same B, different theta
simulation4b <- function(parameters, seed = 1989) {
   set.seed(seed)
  m <- parameters$m
  ave_deg <- 10
  alpha <- 2.5
  # Generate network parameters ------------------------------------------------
  n <- parameters$n
  K <- parameters$K
  Z <- kronecker(diag(K), rep(1, n/K))
  degree_corrections <- lapply(1:m, function(i) {
   if (parameters$degree_distribution=="exp"){
    theta <- rexp(n) + 0.2
  }
  else if (parameters$degree_distribution=="pow") {
     theta <- runif(min=0,max=1,n)^(-1/(alpha-1))
  }
    as.vector(theta / (Z%*%crossprod(Z,theta)/(n/K)))
  })
  B_matrices <- lapply(1:m, function(i) {
    B <- 0.06*diag(K) + 0.04
    B
  })
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  return(list(Adj_list = Adj_list, truecom = as.vector(Z %*% 1:K)))
}






# Simulation 6
### Simulation 6: changing high degree
simulation6 <- function(parameters, seed = 1989) {
  set.seed(seed)
  m <- parameters$m
  ave_deg <- 10
  alpha <- 2.5
  # Generate network parameters ------------------------------------------------
  n <- parameters$n
  K <- parameters$K
  
  ave_deg <- 10
  Z <- kronecker(diag(K), rep(1, n/K))
  
  theta1 <- rep(
  c(
    rep(0.15, floor(0.5*n/K)),
    rep(0.8, ceiling(0.5*n/K))
  ),
  K
)
browser()
  theta2 <- 0.95 - theta1
  theta1 <- as.vector(theta1 / (Z%*%crossprod(Z,theta1)/(n/K)))
  theta2 <- as.vector(theta2 / (Z%*%crossprod(Z,theta2)/(n/K)))               
  degree_corrections <- lapply(1:m, function(i) if(i%%2==1){
    theta1
  }else{
    theta2
  })
  
  B_matrices <- lapply(1:m, function(i) {
    B <- 0.06*diag(K) + 0.04
    B
  })
  
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  #plot_adjmatrix(Adj_list[[1]])
  #colSums(Adj_list[[1]])
  truecom <- as.vector(Z %*% 1:K)
  
  results <- list(Adj_list = Adj_list, truecom = truecom)
  return(results)
}





# Simulation 7
### Simulation 7: changing high degree and random B
simulation7 <- function(parameters, seed = 1989) {
   set.seed(seed)
  m <- parameters$m
  ave_deg <- 10
  alpha <- 2.5
  # Generate network parameters ------------------------------------------------
  n <- parameters$n
  K <- parameters$K
  Z <- kronecker(diag(K), rep(1, n/K))
  
 theta1 <- rep(
  c(
    rep(0.15, floor(0.5*n/K)),
    rep(0.8, ceiling(0.5*n/K))
  ),
  K
)
  theta2 <- 0.95 - theta1
  theta1 <- as.vector(theta1 / (Z%*%crossprod(Z,theta1)/(n/K)))
  theta2 <- as.vector(theta2 / (Z%*%crossprod(Z,theta2)/(n/K)))               
  degree_corrections <- lapply(1:m, function(i) if(i%%2==1){
    theta1
  }else{
    theta2
  })
  
  B_matrices <- lapply(1:m, function(i) {
    p <- runif(1)
    q <- runif(1)
    ((p-q) *diag(K) + q)
  })
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  #plot_adjmatrix(Adj_list[[1]])
  #colSums(Adj_list[[1]])
  truecom <- as.vector(Z %*% 1:K)
  
  results <- list(Adj_list = Adj_list, truecom = truecom)
  return(results)
}

### Simulation 8: core periphery B same theta
simulation8 <- function(parameters, seed = 1989) {
  set.seed(seed)
  m <- parameters[1]
  ave_deg <- 10
  
  # Generate network parameters ------------------------------------------------
  n <- 150
  K <- 2
  Z <- kronecker(diag(K), rep(1, n/K))
  #theta <- runif(n, min = 0.05, max = 1)
  theta <- rexp(n) + 0.2
  theta <- as.vector(theta / (Z%*%crossprod(Z,theta)/(n/K)))
  
  B <- matrix(c(
    1.0, 0.5, 
    0.5, 0.05
  ), nrow = K, byrow = TRUE)
  
  degree_corrections <- lapply(1:m, function(i) theta)
  B_matrices <- lapply(1:m, function(i) B)
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  return(list(Adj_list = Adj_list, truecom = as.vector(Z %*% 1:K)))
}

### Simulation 9: core periphery B, different theta
simulation9 <- function(parameters, seed = 1989) {
  set.seed(seed)
  m <- parameters[1]
  ave_deg <- 10
  # Generate network parameters ------------------------------------------------
  n <- 150
  K <- 3
  Z <- kronecker(diag(K), rep(1, n/K))
  
  degree_corrections <- lapply(1:m, function(i) {
    theta <- runif(n, min = 0.05, max = 1)
    theta <- rexp(n) + 0.2
    as.vector(theta / (Z%*%crossprod(Z,theta)/(n/K)))
  })
  B <- matrix(c(
    1.0, 0.5, 0.2,
    0.5, 0.25, 0.05,
    0.2, 0.05, 0.01
  ), nrow = K, byrow = TRUE)
  
  B_matrices <- lapply(1:m, function(i) B)
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  return(list(Adj_list = Adj_list, truecom = as.vector(Z %*% 1:K)))
}

## Simulation 10: core periphery changing high degree
simulation10 <- function(parameters, seed = 1989) {
  set.seed(seed)
  m <- parameters[1]
  # Set network parameters ------------------------------------------------
  n <- 150
  K <- 3
  
  ave_deg <- 10
  Z <- kronecker(diag(K), rep(1, n/K))
  
  theta1 <- rep(c(rep(0.15, 0.5*n/K), rep(0.8, 0.5*n/K)), K)
  theta2 <- 0.95 - theta1
  theta1 <- as.vector(theta1 / (Z%*%crossprod(Z,theta1)/(n/K)))
  theta2 <- as.vector(theta2 / (Z%*%crossprod(Z,theta2)/(n/K)))               
  degree_corrections <- lapply(1:m, function(i) if(i%%2==1){
    theta1
  }else{
    theta2
  })
  
  B <- matrix(c(
    1.0, 0.5, 0.2,
    0.5, 0,25, 0.05,
    0.2, 0.05, 0.01
  ), nrow = K, byrow = TRUE)
  
  
  # Sample graphs --------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]], degree_corrections[[i]] * Z)
    P <- (ave_deg*n/sum(P)) * P
    sample_from_P(P)
  })
  #plot_adjmatrix(Adj_list[[1]])
  #colSums(Adj_list[[1]])
  truecom <- as.vector(Z %*% 1:K)
  
  results <- list(Adj_list = Adj_list, truecom = truecom)
  return(results)
}

#################
### Simulation C: identifiability
simulation_identifiability_m <- function(parameters, seed = 1989) {
  set.seed(seed)
  m <- parameters[1]
  ave_deg <- 10
  
  # Generate network parameters ------------------------------------------------
  n <- 150
  K <- 3
  Z <- kronecker(diag(K), rep(1, n/K))
  
  theta <- rexp(n) + 0.2
  theta <- as.vector(theta / (Z %*% crossprod(Z, theta) / (n/K)))
  
  # --- Define complementary rank-deficient B matrices (A & B) -----------------
  # collapse 1 & 2
  # B_A <- matrix(c(
  #   1, 1, 0.3,
  #   1, 1, 0.3,
  #   0.3, 0.3, 0.9
  # ), nrow = 3, byrow = TRUE)
  
  # # collapse 2 & 3
  # B_B <- matrix(c(
  #   1, 0.4, 0.4,
  #   0.4, 1, 1,
  #   0.4 , 1, 1
  # ), nrow = 3, byrow = TRUE)
   B_A <- matrix(c(
    0.10, 0.10, 0.04,
    0.10, 0.10, 0.04,
    0.04, 0.04, 0.06
  ), nrow = 3, byrow = TRUE)
  
  # collapse 2 & 3
  B_B <- matrix(c(
    0.11, 0.04, 0.04,
    0.04, 0.09, 0.09,
    0.04, 0.09, 0.09
  ), nrow = 3, byrow = TRUE)
  
  # --- Layer-wise theta --------------------------------------------------------
  degree_corrections <- lapply(1:m, function(i) theta)
  
  # --- New B_matrices: complementary collapse patterns -------------------------
  B_matrices <- lapply(1:m, function(i) {
    if (i %% 2 == 1) B_A else B_B
  })
  
  # Sample graphs ---------------------------------------------------------------
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]],
                    degree_corrections[[i]] * Z)
    P <- (ave_deg * n / sum(P)) * P
    sample_from_P(P)
  })
  
  return(list(Adj_list = Adj_list, truecom = as.vector(Z %*% 1:K)))
}

simulation_identifiability_delta <- function(parameters, seed = 1989) {
  set.seed(seed)
  
  m <- 20
  delta <- parameters[1]  # proportion of A-type collapse layers
  
  ave_deg <- 10
  
  # Generate network parameters ------------------------------------------------
  n <- 150
  K <- 3
  Z <- kronecker(diag(K), rep(1, n/K))
  
  theta <- rexp(n) + 0.2
  theta <- as.vector(theta / (Z %*% crossprod(Z, theta) / (n/K)))
  
  # --- Define complementary rank-deficient B matrices (A & B) -----------------
  # collapse 1 & 2
  B_A <- matrix(c(
    0.10, 0.10, 0.04,
    0.10, 0.10, 0.04,
    0.04, 0.04, 0.06
  ), nrow = 3, byrow = TRUE)
  
  # collapse 2 & 3
  B_B <- matrix(c(
    0.11, 0.04, 0.04,
    0.04, 0.09, 0.09,
    0.04, 0.09, 0.09
  ), nrow = 3, byrow = TRUE)
  
  # --- Layer-wise theta --------------------------------------------------------
  degree_corrections <- lapply(1:m, function(i) theta)
  
  # --- Generate B_matrices based on proportion delta ---------------------------
  # number of A and B layers
  num_A <- round(delta * m)
  num_B <- m - num_A
  
  # randomize the layer order (important!)
  layer_types <- sample(c(rep("A", num_A), rep("B", num_B)))
  
  B_matrices <- lapply(1:m, function(i) {
    if (layer_types[i] == "A") B_A else B_B
  })
  
  # ----------------------------------------------------------------------------- 
  # Sample graphs
  Adj_list <- lapply(1:m, function(i) {
    P <- tcrossprod((degree_corrections[[i]] * Z) %*% B_matrices[[i]],
                    degree_corrections[[i]] * Z)
    P <- (ave_deg * n / sum(P)) * P
    sample_from_P(P)
  })
  
  return(list(Adj_list = Adj_list, truecom = as.vector(Z %*% 1:K)))
}
