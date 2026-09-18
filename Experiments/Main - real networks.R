
library(Matrix)
library(aricode)
library(reticulate)
source("R/comdetmethods.R")
library(igraph)
library(mclust)
library(multinet)

#NOTE: To run mFROST and graph-tool Python must also be installed, together with the dependencies listed in the requirements.txt, 
# Configure reticulate to use the Python environment containing these dependencies before running the experiments:
# Modify the file Experiments/R/comdetmethods to use mFROST in Python

####################### Load DATA ##########################################################

build_AUCS <- function(edge_file, node_file){


  edges <- read.csv(
    edge_file,
    header = TRUE,
    stringsAsFactors = FALSE
  )
  
  nodes <- read.csv(
    node_file,
    header = TRUE,
    stringsAsFactors = FALSE
  )

  node_names <- nodes$node
  N <- length(node_names)
  
  node_index <- setNames(1:N, node_names)

  layers <- unique(edges$layer)
  
  adjacency_list <- list()


  for(layer_name in layers){
    
    A <- Matrix(
      0,
      nrow = N,
      ncol = N,
      sparse = TRUE
    )
    
    rownames(A) <- node_names
    colnames(A) <- node_names
    
    
    e <- edges[edges$layer == layer_name, ]
    
    
    for(i in 1:nrow(e)){
      
      s <- e$source[i]
      t <- e$target[i]
      
      A[node_index[s], node_index[t]] <- 1
      A[node_index[t], node_index[s]] <- 1
    }
    
    
    adjacency_list[[layer_name]] <- A
  }
  names(adjacency_list) <- NULL

  groups <- as.numeric(sub("G", "", nodes$group))
 
  return(
    list(
      A = adjacency_list,
      labels = groups
    )
  )
}

build_cora_multilayer <- function(content_path, cites_path, k = 20) {
  

  # Load content file
  content <- read.table(content_path,
                        header = FALSE,
                        stringsAsFactors = FALSE)
  
  paper_id <- content[, 1]
  labels_raw <- content[, ncol(content)]
  
  X <- as.matrix(content[, 2:(ncol(content) - 1)])
  rownames(X) <- paper_id
  
 
  # KEEP ONLY 3 CLASSES
  keep_classes <- c(
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods"
  )
  
  keep_idx <- which(labels_raw %in% keep_classes)
  
  paper_id <- paper_id[keep_idx]
  labels_raw <- labels_raw[keep_idx]
  X <- X[keep_idx, ]
  
  # convert labels to 1..3
  labels <- match(labels_raw, keep_classes)
  
  n <- length(paper_id)
  

  # CITATION LAYER
  cites <- read.table(cites_path,
                      header = FALSE,
                      stringsAsFactors = FALSE)
  
  colnames(cites) <- c("cited", "citing")
  
  # filter edges to kept nodes
paper_id <- as.character(paper_id)
cites$citing <- as.character(cites$citing)
cites$cited <- as.character(cites$cited)

adj_citation <- matrix(0, n, n,
                       dimnames = list(paper_id, paper_id))

valid <- cites$citing %in% paper_id &
         cites$cited %in% paper_id

cites <- cites[valid, ]

adj_citation[cbind(cites$citing, cites$cited)] <- 1
adj_citation[cbind(cites$cited, cites$citing)] <- 1

  # SIMILARITY LAYER

  norm_X <- sqrt(rowSums(X^2))
  norm_X[norm_X == 0] <- 1  # avoid division by zero
  
  X_norm <- X / norm_X
  
  sim <- X_norm %*% t(X_norm)
  diag(sim) <- 0
  
 
  # kNN GRAPH

  adj_similarity <- matrix(0, n, n)
  rownames(adj_similarity) <- paper_id
  colnames(adj_similarity) <- paper_id
  
  for (i in 1:n) {
    top_k <- order(sim[i, ], decreasing = TRUE)[1:min(k, n)]
    adj_similarity[i, top_k] <- 1
  }
  
  # symmetrize
 adj_similarity <- pmax( adj_similarity, t( adj_similarity))
  

  # RETURN

    A <- list(
  adj_citation,
  adj_similarity)
  return(list(
    A = A ,
    labels           = labels
  ))
}

build_citeseer_multilayer <- function(content_path, cites_path, k = 20) {
  

  # Load content file
  content <- read.table(content_path,
                        header = FALSE,
                        stringsAsFactors = FALSE)
  
  paper_id <- content[, 1]
  labels_raw <- content[, ncol(content)]
  
  X <- as.matrix(content[, 2:(ncol(content) - 1)])
  rownames(X) <- paper_id
  
  classes <- c(
    "Agents",
			"AI",
			"DB",
			"IR",
			"ML",
			"HCI"
  )
  # convert labels to 1..3
  labels <- match(labels_raw, classes)
  
  n <- length(paper_id)
  

  # CITATION LAYER

  cites <- read.table(cites_path,
                      header = FALSE,
                      stringsAsFactors = FALSE)
  
  colnames(cites) <- c("cited", "citing")
  
  # filter edges to kept nodes
paper_id <- as.character(paper_id)
cites$citing <- as.character(cites$citing)
cites$cited <- as.character(cites$cited)

adj_citation <- matrix(0, n, n,
                       dimnames = list(paper_id, paper_id))

valid <- cites$citing %in% paper_id &
         cites$cited %in% paper_id

cites <- cites[valid, ]

adj_citation[cbind(cites$citing, cites$cited)] <- 1
adj_citation[cbind(cites$cited, cites$citing)] <- 1

  #  SIMILARITY LAYER

  norm_X <- sqrt(rowSums(X^2))
  norm_X[norm_X == 0] <- 1  # avoid division by zero
  
  X_norm <- X / norm_X
  
  sim <- X_norm %*% t(X_norm)
  diag(sim) <- 0

  # kNN GRAPH

  adj_similarity <- matrix(0, n, n)
  rownames(adj_similarity) <- paper_id
  colnames(adj_similarity) <- paper_id
  
  for (i in 1:n) {
    top_k <- order(sim[i, ], decreasing = TRUE)[1:min(k, n)]
    adj_similarity[i, top_k] <- 1
  }
  
  # symmetrize
   adj_similarity <- pmax( adj_similarity, t( adj_similarity))
  
 
  # RETURN


  A <- list(
  adj_citation,
  adj_similarity)
  return(list(
    A = A ,
    labels           = labels
  ))
}

build_caltech <- function(labels_file, edges_file) {
  
  # Read labels 
  labels_data <- read.table(
    labels_file,
    header = FALSE
  )
  
  # colonne 1 = node id, colonne 2 = label
  labels <- labels_data$V2
  
  n <- length(labels)
  
  
  # Read edges
  edges <- read.table(
    edges_file,
    header = FALSE
  )
  
  # colonnes :
  # V1 = layer
  # V2 = node i
  # V3 = node j
  
  num_layers <- max(edges$V1)
  
  A_list <- vector("list", num_layers)
  
  
  # Build adjacency matrices
  for (l in 1:num_layers) {
    
    edges_l <- edges[edges$V1 == l, ]
    
    A <- sparseMatrix(
      i = c(edges_l$V2, edges_l$V3),
      j = c(edges_l$V3, edges_l$V2),
      x = 1,
      dims = c(n, n)
    )
    
    
    A[A > 1] <- 1
    
    A_list[[l]] <- A
  }
  
  
  return(
    list(
      labels = labels,
      A = A_list
    )
  )
}

##################### Methods #############################################################
run_all_methods <- function(Adj_list, truecoms) {
  set.seed(42)
  idx <- !is.na(truecoms)
  K <- length(unique(truecoms[idx]))
  
   methods_to_run <- c("graph-tool","mfrost","dcmase","ave_spherical",
                       "sq-bias-adjusted","mase-spherical")
 
  results <- lapply(methods_to_run, function(method) {
    print(method)
    
    labels <- allmethods(Adj_list, K, method = method)
    
    res<-list(
      NMI = NMI(truecoms[idx], as.vector(labels)[idx],  variant = "sum"),
      errorRate = classError(labels[idx], truecoms[idx])$errorRate,
      ARI = ARI(truecoms[idx], as.vector(labels)[idx])
    )
    print(res)
  })
  
  # names(results) <- c("graph-tool","FROST_MF","FROST_US","FROST_DCMASE",
  #                     "US","MF","OLMF","DC_MASE","Sum A",
  #                     "S-A^2-Bias-adj","MASE")
  names(results) <- c("graph-tool","mfrost","dcmase","ave_spherical",
                       "sq-bias-adjusted","mase-spherical")
  return(results)
}

library(igraph)

multilayer_properties <- function(adj_list) {
  
  L <- length(adj_list)
  N <- nrow(adj_list[[1]])
  
  cat("Number of nodes:", N, "\n")
  cat("Number of layers:", L, "\n\n")
  
  
  # Statistics for each layer
  
  layer_statistics <- data.frame(
    layer = 1:L,
    edges = NA_real_,
    density = NA_real_,
    avg_degree = NA_real_,
    degree_sd = NA_real_,
    degree_cv = NA_real_,
    max_degree = NA_real_,
    isolated_nodes = NA_real_,
    clustering = NA_real_,
    components = NA_real_
  )
  
  degrees <- vector("list", L)
  
  for (l in 1:L) {
    
    A <- adj_list[[l]]
    
    
    diag(A) <- 0
    
    g <- igraph::graph_from_adjacency_matrix(
      A,
      mode = "undirected",
      diag = FALSE
    )
    
    deg <- igraph::degree(g)
    degrees[[l]] <- deg
    
    # Number of edges
    layer_statistics$edges[l] <- igraph::ecount(g)
    
    # Density
    layer_statistics$density[l] <- igraph::edge_density(g)
    
    # Degree statistics
    layer_statistics$avg_degree[l] <- mean(deg)
    layer_statistics$degree_sd[l] <- sd(deg)
    
    # Coefficient of variation
    if (mean(deg) > 0) {
      layer_statistics$degree_cv[l] <- sd(deg) / mean(deg)
    } else {
      layer_statistics$degree_cv[l] <- NA
    }
    
    layer_statistics$max_degree[l] <- max(deg)
    
    # Proportion of isolated nodes
    layer_statistics$isolated_nodes[l] <- mean(deg == 0)
    
    # Global clustering coefficient
    layer_statistics$clustering[l] <- igraph::transitivity(
      g,
      type = "global"
    )
    
    # Number of connected components
    layer_statistics$components[l] <- igraph::components(g)$no
  }
  
 
  # Summary across layers

  summary <- data.frame(
    measure = c(
      "Number of edges",
      "Density",
      "Average degree",
      "Degree SD",
      "Degree CV",
      "Maximum degree",
      "Isolated nodes (%)",
      "Clustering coefficient",
      "Number of components"
    ),
    
    mean = c(
      mean(layer_statistics$edges),
      mean(layer_statistics$density),
      mean(layer_statistics$avg_degree),
      mean(layer_statistics$degree_sd),
      mean(layer_statistics$degree_cv, na.rm = TRUE),
      mean(layer_statistics$max_degree),
      100 * mean(layer_statistics$isolated_nodes),
      mean(layer_statistics$clustering, na.rm = TRUE),
      mean(layer_statistics$components)
    ),
    
    sd = c(
      sd(layer_statistics$edges),
      sd(layer_statistics$density),
      sd(layer_statistics$avg_degree),
      sd(layer_statistics$degree_sd),
      sd(layer_statistics$degree_cv, na.rm = TRUE),
      sd(layer_statistics$max_degree),
      100 * sd(layer_statistics$isolated_nodes),
      sd(layer_statistics$clustering, na.rm = TRUE),
      sd(layer_statistics$components)
    )
  )
  

  paper_summary <- data.frame(
    property = c(
      "Nodes",
      "Layers",
      "Edges",
      "Density",
      "Average degree",
      "Degree CV",
      "Max degree",
      "Isolated nodes (%)",
      "Clustering",
      "Components"
    ),
    
    value = c(
      N,
      L,
      sprintf("%.1f",
              sum(layer_statistics$edges)),
      sprintf("%.4f ± %.4f",
              mean(layer_statistics$density),
              sd(layer_statistics$density)),
      sprintf("%.2f ± %.2f",
              mean(layer_statistics$avg_degree),
              sd(layer_statistics$avg_degree)),
      sprintf("%.2f ± %.2f",
              mean(layer_statistics$degree_cv, na.rm = TRUE),
              sd(layer_statistics$degree_cv, na.rm = TRUE)),
      sprintf("%.1f ± %.1f",
              mean(layer_statistics$max_degree),
              sd(layer_statistics$max_degree)),
      sprintf("%.2f ± %.2f",
              100 * mean(layer_statistics$isolated_nodes),
              100 * sd(layer_statistics$isolated_nodes)),
      sprintf("%.3f ± %.3f",
              mean(layer_statistics$clustering, na.rm = TRUE),
              sd(layer_statistics$clustering, na.rm = TRUE)),
      sprintf("%.1f ± %.1f",
              mean(layer_statistics$components),
              sd(layer_statistics$components))
    )
  )
  

  # Similarity between layers
 
  
  cat("\n---- Layer similarity ----\n")
  
  similarity <- NULL
  average_similarity <- NA
  
  if (L > 1) {
    
    similarity <- matrix(
      0,
      nrow = L,
      ncol = L
    )
    
    for (i in 1:L) {
      for (j in 1:L) {
        
        Ai <- adj_list[[i]]
        Aj <- adj_list[[j]]
        
        # Binary similarity between adjacency matrices
        similarity[i, j] <- mean(Ai == Aj)
      }
    }
    
    average_similarity <- mean(
      similarity[upper.tri(similarity)]
    )
    
    print(round(similarity, 3))
    
    cat(
      "\nAverage layer similarity:",
      round(average_similarity, 3),
      "\n"
    )
  }
  
  
  # Print results
 
  
  cat("\n---- Layer statistics ----\n")
  print(layer_statistics)
  
  cat("\n---- Summary across layers ----\n")
  print(summary)
  
  cat("\n---- Compact summary for paper ----\n")
  print(paper_summary)
  

  # Return

  
  return(
    list(
      layer_statistics = layer_statistics,
      summary = summary,
      paper_summary = paper_summary,
      degrees = degrees,
      layer_similarity = similarity,
      average_layer_similarity = average_similarity
    )
  )
}



######################################## RESULTS ############################

#data <- build_cora_multilayer("Data/cora/cora.content", "Data/cora/cora.cites")
#data <- build_citeseer_multilayer("Data/citeseer/citeseer.content", "Data/citeseer/citeseer.cites")
data <- build_AUCS("Data/AUCS/aucs_edgelist.txt","Data/AUCS/aucs_nodelist.txt")
#data <- build_caltech("Data/caltech_all/labels.txt","Data/caltech_all/edges.txt")
#data <- build_caltech("Data/caltech_all/labels.txt","Data/caltech_all/edges.txt")

#properties <- multilayer_properties(data$A)
results<-run_all_methods(data$A,data$labels)
