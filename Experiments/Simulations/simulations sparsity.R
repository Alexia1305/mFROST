




#########################################################################
### Simulation 1: same B same theta
simulation1s <- function(parameters, seed = 1989) {
  set.seed(seed)
  ave_deg <- parameters[1]
  m <- 20
  
  # Generate network parameters ------------------------------------------------
  n <- 150
  K <- 3
  Z <- kronecker(diag(K), rep(1, n/K))
  #theta <- runif(n, min = 0.05, max = 1)
  theta <- rexp(n) + 0.2
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
simulation2s <- function(parameters, seed = 1989) {
  set.seed(seed)
  ave_deg <- parameters[1]
  m <- 20
  
  # Generate network parameters ------------------------------------------------
  n <- 150
  K <- 3
  Z <- kronecker(diag(K), rep(1, n/K))
  
  #theta <- runif(n, min = 0.05, max = 1)
  theta <- rexp(n) + 0.2
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
simulation3s <- function(parameters, seed = 1989) {
  set.seed(seed)
  ave_deg <- parameters[1]
  m <- 20
  # Generate network parameters ------------------------------------------------
  n <- 150
  K <- 3
  Z <- kronecker(diag(K), rep(1, n/K))
  #degree_corrections <- lapply(1:m, function(i) runif(n, min = sqrt(0.05), max = 1)^2)
  degree_corrections <- lapply(1:m, function(i) {
    theta <- runif(n, min = 0.05, max = 1)
    theta <- rexp(n) + 0.2
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
simulation4s <- function(parameters, seed = 1989) {
  set.seed(seed)
  ave_deg <- parameters[1]
  m <- 20
  # Generate network parameters ------------------------------------------------
  n <- 150
  K <- 3
  Z <- kronecker(diag(K), rep(1, n/K))
  
  degree_corrections <- lapply(1:m, function(i) {
    theta <- runif(n, min = 0.05, max = 1)
    theta <- rexp(n) + 0.2
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
simulation6s <- function(parameters, seed = 1989) {
  set.seed(seed)
  ave_deg <- parameters[1]
  # Set network parameters ------------------------------------------------
  n <- parameters[2] #150
  K <- parameters[3] #3
  
  m <- 20
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
simulation7s <- function(parameters, seed = 1989) {
  set.seed(seed)
  ave_deg <- parameters[1]
  # Set network parameters ------------------------------------------------
  n <- parameters[2] #150
  K <- parameters[3] #3
  
  m <- 20
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


