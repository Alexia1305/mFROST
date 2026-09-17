#'
#' Function to perform degree-corrected multiple adjacency spectral embedding (DC-MASE)
#'
#' @param Adj_list a list of adjacency matrices with the same size n x n
#' @param d number of joint embedding dimensions. If NA, dimension is chosen automatically via scree plot.
#' @param d_vec vector of individual embedding dimensions for each graph. If NA, this is the same as d.
#' @param scaled whether to use scaled or unscaled ASE.
#' @param row.normalization parameter that indicates the option to normalize the rows of each embedding. Options: "2" for spherical spectral clustering, "score" for SCORE method by Jin (2015)
#' @param diag.augment logical. When true, the diagonal augmentation method for ASE is performed.
#' @param elbow_graph number of elbow selected in Zhu & Ghodsi method for the scree plot of each individual graph's adjacency matrix eigenvalues.
#' @param elbow_dcmase number of elbow selected in Zhu & Ghodsi method for the scree plot of the singular values of the concatenated spectral embeddings of DC-MASE.
#' @param show.scree.results when TRUE, the histogram of the estimated d for each graph, and the scree plot of the singular values of  the graph is shown if d is not specified.
#' @param par whether to run in parallel (TRUE/FALSE)
#' @param numpar number of clusters for parallel implementation
#'
#' @return A matrix of size n x d, with the degree corrected joint embedding.
#'
#' @references
#'
#' @author Jes\'us Arroyo <jarroyo@tamu.edu>
#' @export
dcmase <- function(Adj_list, d = NA, d_vec = NA, scaled = TRUE,
                    row.normalization = "2", diag.augment = FALSE,
                   elbow_graph = 1, elbow_dcmase = 2,
                   show.scree.results = FALSE,
                   par = FALSE, numpar = 12) {
  if(is.na(d_vec)) {
    d_vec = rep(d, length(Adj_list))
  }
  # running in parallel--------------------------------
  if(par) {
    stop("Not yet implemented")
    cl <- parallel::makeCluster(numpar)
    parallel::clusterEvalQ(cl, source("R/loadAll.R"))
    parallel::clusterExport(cl = cl, varlist = list("g.ase", "eig_embedding", "getElbows", "Adj_list", "scaled",
                                                    "elbow_graph", "d_vec", "diag.augment"), envir = environment())
    if(row.normalization=="2") {
      latpos.list <- parallel::parLapply(cl = cl, 1:length(Adj_list), function(i) {
        if(scaled) {
          U <- g.ase(Adj_list[[i]], d = d_vec[i], diag.augment = diag.augment, elbow = elbow_graph)$X
        }else{
          U <- eig_embedding(Adj_list[[i]], d = d_vec[i], diag.augment = diag.augment, elbow = elbow_graph) 
        }
        unorms <- apply(U, 1, function(x) sqrt(sum(x^2)))
        U <- U / ifelse(unorms < 1e-15, 1, unorms)
        U
      })
    }else{
      if(row.normalization == "score") {
        latpos.list <- parallel::parLapply(cl = cl,1:length(Adj_list), function(i) {
          if(d_vec[i] <2) {stop("Wrong dimensions selected")}
          U <- eig_embedding(Adj_list[[i]], d = d_vec[i], diag.augment = diag.augment, elbow = elbow_graph) 
          V <- U[, 2:d_vec[i], drop = FALSE] /  ifelse(abs(U[,1]) < 1e-15, 1, U[,1])
          V
        })
      }else{
        parallel::stopCluster(cl)
        stop("Normalization option not available") 
      }
    }
     parallel::stopCluster(cl)
  } else { # not parallel ----------------------------------
    if(row.normalization=="2") {
      latpos.list <- lapply(1:length(Adj_list), function(i) {
        if(scaled) {
          U <- g.ase(Adj_list[[i]], d = d_vec[i], diag.augment = diag.augment, elbow = elbow_graph)$X
        }else{
          U <- eig_embedding(Adj_list[[i]], d = d_vec[i], diag.augment = diag.augment, elbow = elbow_graph) 
        }
        unorms <- apply(U, 1, function(x) sqrt(sum(x^2)))
        U <- U / ifelse(unorms < 1e-15, 1, unorms)
        U
      })
    }else{
      if(row.normalization == "score") {
        latpos.list <- lapply(1:length(Adj_list), function(i) {
          if(d_vec[i] <2) {stop("Wrong dimensions selected")}
          U <- eig_embedding(Adj_list[[i]], d = d_vec[i], diag.augment = diag.augment, elbow = elbow_graph) 
          V <- U[, 2:d_vec[i], drop = FALSE] /  ifelse(abs(U[,1]) < 1e-15, 1, U[,1])
          V
        })
      }else{
        stop("Normalization option not available") 
      }
    }
  }
  V_all  <- Reduce(cbind, latpos.list)
  jointsvd <- svd(V_all)
  if(show.scree.results) {
    plot(jointsvd$d, main = "Singular values of matrix of concatenated embeddings")
  }
  if(is.na(d)) {
    if(show.scree.results) {
      hist(sapply(latpos.list, ncol), main = "Estimated d for each graph")
    }
    d = getElbows(jointsvd$d, plot = show.scree.results)[elbow_dcmase]
  }
  if(row.normalization == "2"){
    V = jointsvd$u[, 1:d, drop = FALSE] 
  }else{ if(row.normalization == "score") {
      V = jointsvd$u[, 1:(d-1), drop = FALSE]  
    }
  }
  return(V)
}
