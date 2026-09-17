library(ggplot2)


# Plot time series of degree corrections
plot_degree_corrections <- function(comdet_res) {
  thetas = comdet_res$degree_corrections
  colnames(thetas) = 1:ncol(thetas)
  df_thetas <- data.frame(Community = comdet_res$community_memberships,
                          vertex = 1:nrow(thetas),
                          thetas = thetas)
  colnames(df_thetas)[-c(1,2)] <-  1:ncol(thetas)
  library(reshape2)
  
  df_melted <- melt(df_thetas, id.vars = c("Community", "vertex"))
  
  df_melted$Community <- factor(df_melted$Community)
  df_melted$vertex <- factor(df_melted$vertex)
  df_melted$variable <- as.numeric(df_melted$variable)
  df_melted$date <- as.Date(paste(floor((df_melted$variable-1)/12)+2016, "-", 
                                  ((df_melted$variable-1)%%12) + 1, "-", "1", sep = ""))
  library(ggplot2)
  ggplot(df_melted, aes(x = date, y = value, group = vertex, col = Community)) +
    geom_vline(aes(xintercept = as.Date("2020-01-01")), linetype = 2) +
    geom_line() +
    facet_wrap(df_melted$Community, scales = "free") +
    theme_bw() + xlab("Time") + ylab("degree corrections") +
    theme(legend.position="none")
    #scale_x_date(date_breaks = "1 month", date_minor_breaks = "1 week",
    #             date_labels = "%B")
}


# Plot time series of block connectivity matrices
plot_pairs_Bmats_diag_normalized  <- function(comdet_res) {
  K <- max(comdet_res$community_memberships)
  df = list()
  m <- ncol(comdet_res$degree_corrections)
  B_mats <- comdet_res$B_matrices
  for(i in 1:m) {
    B_mats[i, , ] <- diag(1/sqrt(diag(B_mats[i, ,])))  %*% B_mats[i, ,] %*% diag(1/sqrt(diag(B_mats[i, ,])))
  }
  for(i in 1:K){
    for(j in 1:K) {
      df <- rbind(df, data.frame(B = B_mats[,i,j], T = 1:m, com_1 = i, com_2 = j))
    }
  }
  
  library(ggplot2)
  library(reshape2)
  ggplot(df, aes(T, B)) + geom_line() + facet_grid(com_1 ~ com_2, scales = "free_y") +
    geom_vline(xintercept = 51, col = "red")
  
}

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
plot_pairs_Bmats  <- function(comdet_res) {
  K <- max(comdet_res$community_memberships)
  df = list()
  m <- ncol(comdet_res$degree_corrections)
  for(i in 1:K){
    for(j in 1:K) {
      df <- rbind(df, data.frame(B = comdet_res$B_matrices[,i,j], T = 1:m, com_1 = i, com_2 = j))
    }
  }
  
  df$date <- as.Date(paste(floor((df$T-1)/12)+2016, "-", 
                                  ((df$T-1)%%12) + 1, "-", "1", sep = ""))
  
  library(ggplot2)
  library(reshape2)
  library(grid)
  p <- ggplot(df, aes(date, B)) + geom_line() + facet_grid(com_1 ~ com_2, scales = "free_y") +
    geom_vline(aes(xintercept = as.Date("2020-01-01")), linetype = 2) +
    geom_vline(xintercept = 51, col = "red") +
    theme_bw() + xlab("Time") + ylab("Block connectivity parameters") 
  
  g <- ggplot_gtable(ggplot_build(p))
  strip_both <- which(grepl('strip-', g$layout$name))
  fills <- c(gg_color_hue(K), gg_color_hue(K))
  k <- 1
  for (i in strip_both) {
    j <- which(grepl('rect', g$grobs[[i]]$grobs[[1]]$childrenOrder))
    g$grobs[[i]]$grobs[[1]]$children[[j]]$gp$fill <- fills[k]
    k <- k+1
  }
  grid.draw(g)
  
}








plot_usmap <- function(communities, title = NA) {
  # Plot community detection result 
  # https://jtr13.github.io/cc19/different-ways-of-plotting-u-s-map-in-r.html
  
  #install.packages("mapdata")
  require(maps)
  require(mapdata)
  
  coordinates <- data.frame(airport_coordinates, com = factor(communities))
  colnames(coordinates) <- c("x", "y", "Community")
  usa <- map_data('usa')
  state <- map_data("state")
  
  ggplot() + 
    geom_polygon(data=state, aes(x=long, y=lat,  group=group), color = "gray", fill = "white") + 
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank(),
          axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank()) + 
    coord_fixed(1.3) +
    geom_point(data= coordinates, aes(x = x,y = y, fill = Community, 
                                      shape = Community)) +
    scale_shape_manual(values = c(21:24, 5:14)) +
    theme(legend.position="top", legend.text.align = 0)
}
plot_degree_corrections_one_community <- function(comdet_res, com = 1, select_vertices = NA) {
  require(ggthemes)
  thetas = comdet_res$degree_corrections
  colnames(thetas) = 1:ncol(thetas)
  
  df_thetas <- data.frame(Community = comdet_res$community_memberships,
                          vertex = 1:nrow(thetas),
                          thetas = thetas, highlight = 0.5*(airport_names %in% select_vertices) + 0.5,
                          Airport = ifelse(airport_names %in% select_vertices, airport_names, "Other"))
  colnames(df_thetas)[-c(1,2, 72:73)] <-  1:ncol(thetas)
  library(reshape2)
  
  df_melted <- melt(df_thetas, id.vars = c("Community", "vertex", "highlight", "Airport"))
  
  df_melted$Community <- factor(df_melted$Community)
  df_melted$vertex <- factor(df_melted$vertex)
  df_melted$Airport <- factor(df_melted$Airport)
  df_melted$variable <- as.numeric(df_melted$variable)
  df_melted$date <- as.Date(paste(floor((df_melted$variable-1)/12)+2016, "-", 
                                  ((df_melted$variable-1)%%12) + 1, "-", "1", sep = ""))
  require(ggplot2)
  require(dplyr)
  df_melted %>%
    filter(Community == com) %>%
    ggplot(aes(x = date, y = value, group = vertex, col = Airport)) +
    geom_vline(aes(xintercept = as.Date("2020-01-01")), linetype = 2) +
    geom_line(aes(alpha=highlight)) +
    scale_colour_manual(values = c(colorblind_pal()(length(select_vertices)+1)[-5], "gray")) +
    #facet_wrap(df_melted$Community, scales = "free") +
    theme_bw() + xlab("Time") + ylab("degree corrections") +
    scale_alpha_continuous(range = c(0.3,1)) + 
    guides(alpha = "none")
}
