
data_summary <- function(data, varname, groupnames) {
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE),
      se = sd(x[[col]], na.rm=TRUE) / sqrt(length(x[[col]])))
  }
  data_sum <- ddply(data, groupnames, .fun=summary_func,
                    varname)
  data_sum <- plyr::rename(data_sum, c("mean" = varname))
  return(data_sum)
}

make_ggplot_res <- function(results, parameter_name, xbreaks = c(1, 5, 10, 15, 20, 25)) {
  require(ggplot2)
  require(reshape2)
  require(scales)
  results_melted <- melt(results, id.vars = "parameter")
  names(results_melted) <- c("parameter", "Method", "ARI")
  
  resdf = data_summary(results_melted, "ARI", c("parameter", "Method"))
  
  p<- ggplot(resdf, aes(x=parameter, y = ARI)) + 
    geom_line(aes(color=Method, linetype=Method)) +
    geom_point(aes(color=Method, shape=Method))+
    #geom_errorbar(aes(ymin=ARI-2*se, ymax=ARI+2*se, color = Method), width=0.2) +
    ylim(c(0,1)) +
    scale_x_continuous(breaks= xbreaks, limits = c(1, NA)) +
    xlab(parameter_name) +theme_bw()
  p
}

make_ggplot_res_error <- function(results, parameter_name, xbreaks = c(1, 5, 10, 15, 20, 25),
                                  ylim = c(0,0.75)) {
  require(ggplot2)
  require(reshape2)
  require(scales)
  results_melted <- melt(results, id.vars = "parameter")
  names(results_melted) <- c("parameter", "Method", "Misclustering.error")
  
  resdf = data_summary(results_melted, "Misclustering.error", c("parameter", "Method"))
  
  p<- ggplot(resdf, aes(x=parameter, y = Misclustering.error)) + 
    geom_line(aes(color=Method, linetype=Method)) +
    geom_point(aes(color=Method, shape=Method))+
    #geom_errorbar(aes(ymin=ARI-2*se, ymax=ARI+2*se, color = Method), width=0.2) +
    ylim(ylim) +
    scale_x_continuous(breaks= xbreaks, limits = c(min(1, xbreaks), NA)) +
    xlab(parameter_name) +theme_bw()
  p
}

make_ggplot_multiple <- function(different_scenarios, parameter_name, xbreaks = c(1, 5, 10, 15, 20, 25)) {
  require(ggplot2)
  require(reshape2)
  require(scales)
  results_melted <- melt(different_scenarios , id.vars = c("parameter", "scenario"))
  names(results_melted) <- c("parameter", "Scenario", "Method", "ARI")
  
  resdf = data_summary(results_melted, "ARI", c("parameter", "Method", "Scenario"))
  
  p<- ggplot(resdf, aes(x=parameter, y = ARI)) + 
    geom_line(aes(color=Method, linetype=Method)) +
    geom_point(aes(color=Method, shape=Method))+
    #geom_errorbar(aes(ymin=ARI-2*se, ymax=ARI+2*se, color = Method), width=0.2) +
    ylim(c(0,1)) +
    scale_x_continuous(breaks= xbreaks) +
    xlab(parameter_name) +theme_bw() +
    facet_wrap("Scenario")
  p
}


make_ggplot_multipleBT <- function(different_scenarios, parameter_name, xbreaks = c(1, 5, 10, 15, 20, 25),
                                   methodnames = c("FROST_MF","FROST_US","MF","US","DC-MASE", "Sum A", "Sum A^2 bias adj.", "MASE", "OLMF", "graph-tool")) {
  require(ggplot2)
  require(reshape2)
  require(scales)
  require(ggthemes)
  results_melted <- melt(different_scenarios , id.vars = c("parameter", "scenarioB", "scenarioT"), measure.vars = 1:6)
  names(results_melted) <- c("parameter", "ScenarioB", "ScenarioT", "Method", "ARI")
  
  resdf = data_summary(results_melted, "ARI", c("parameter", "Method", "ScenarioB", "ScenarioT"))
  
  p<- ggplot(resdf, aes(x=parameter, y = ARI)) + 
    geom_line(aes(color=Method, linetype=Method)) +
    geom_point(aes(color=Method, shape=Method))+
    #geom_errorbar(aes(ymin=ARI-2*se, ymax=ARI+2*se, color = Method), width=0.2) +
    ylim(c(0,1)) +
    scale_x_continuous(breaks= xbreaks) +
    xlab(parameter_name) +theme_bw() +
    facet_grid(ScenarioB ~ ScenarioT) +
    scale_color_manual(labels = methodnames, 
                       values = colorblind_pal()(8)[c(7, 2, 4, 6, 3, 8)]) +
    scale_shape_manual(labels = methodnames, 
                       values = c(19, 17,15, 7, 3, 8)) +
    scale_linetype_manual(labels = methodnames, 
                          values = c(1:6)) +
    theme(legend.position="top", legend.text.align = 0)
  p
}

make_ggplot_multipleBT2 <- function(different_scenarios, 
                                    parameter_name, 
                                    xbreaks = c(1, 5, 10, 15, 20, 25),
                                    metric = "ARI",
                                    methodnames = c("mFROST","DC-MASE", "Sum A", "Bias-adjusted SoS", "MASE", "OLMF", "graph-tool","OLMF"),
                                    ylim = c(0,1)) {
  # Packages nécessaires
  require(ggplot2)
  require(reshape2)
  require(scales)
  require(ggthemes)
  
  # Ne garder que les colonnes existantes
  methodnames_valid <- intersect(methodnames, colnames(different_scenarios))
  if(length(methodnames_valid) == 0){
    stop("Aucune colonne valide trouvée dans different_scenarios pour les méthodes fournies.")
  }
  
  # Transformation en format long
  results_melted <- melt(
    different_scenarios,
    id.vars = c("parameter", "scenarioB", "scenarioT"),
    measure.vars = methodnames_valid
  )
  names(results_melted) <- c("parameter", "ScenarioB", "ScenarioT", "Method", "ARI")
  
  # Résumé des données
  resdf <- data_summary(results_melted, "ARI", c("parameter", "Method", "ScenarioB", "ScenarioT"))
  
  # Adaptation automatique des couleurs, formes et linetypes
  n_methods <- length(methodnames_valid)
  colors <- c(
    "#000000", # noir
    "#E69F00", # orange
    "#56B4E9", # bleu ciel
    "#009E73", # vert
    "#F0E442", # jaune
    "#0072B2", # bleu
    "#D55E00", # vermillon
    "#CC79A7", # violet
    "#999999", # gris
    "#A65628"  # brun
  )[1:n_methods]          # Palette adaptée daltoniens
  shapes <- c(
    16, # cercle plein
    17, # triangle plein
    15, # carré plein
    18, # diamant plein
    3,  # plus
    4,  # croix
    8,  # étoile
    0,  # carré vide
    1,  # cercle vide
    2   # triangle vide
  )[1:n_methods]     # Formes adaptées
  linetypes <- c(
    "solid",
    "dashed",
    "dotted",
    "dotdash",
    "longdash",
    "twodash",
    "solid",
    "dashed",
    "dotted",
    "dotdash"
  )[1:n_methods]                            # Types de ligne
  
  # Création du plot
  p <- ggplot(resdf, aes(x = parameter, y = ARI)) +
    geom_line(aes(color = Method, linetype = Method)) +
    geom_point(aes(color = Method, shape = Method)) +
    # geom_errorbar(aes(ymin=ARI-2*se, ymax=ARI+2*se, color = Method), width=0.2) +
    ylim(ylim) +
    scale_x_continuous(breaks = xbreaks) +
    ylab("Misclustering error") +
    xlab(parameter_name) +
    theme_bw() +
    facet_grid(ScenarioB ~ ScenarioT) +
    scale_color_manual(labels = methodnames_valid, values = colors) +
    scale_shape_manual(labels = methodnames_valid, values = shapes) +
    scale_linetype_manual(labels = methodnames_valid, values = linetypes) +
    theme(legend.position = "top", legend.text.align = 0, panel.spacing = unit(0.5, "cm"), legend.title = element_blank(),axis.text = element_text(size = 10),axis.title=element_text(size=11),
  legend.text = element_text(size = 11),strip.text = element_text(size = 11))
  
  return(p)
}

make_ggplot_multipleBT2V <- function(different_scenarios, 
                                    parameter_name, 
                                    xbreaks = c(1, 5, 10, 15, 20, 25),
                                    metric = "ARI",
                                    methodnames = c("FROST_MF","FROST_US","MF","US","DC-MASE", "Sum A", "Sum A^2 bias adj.", "MASE", "OLMF", "graph-tool","LMFO"),
                                    ylim = c(0,1)) {
  # Packages nécessaires
  require(ggplot2)
  require(reshape2)
  require(scales)
  require(ggthemes)
  
  # Ne garder que les colonnes existantes
  methodnames_valid <- intersect(methodnames, colnames(different_scenarios))
  if(length(methodnames_valid) == 0){
    stop("Aucune colonne valide trouvée dans different_scenarios pour les méthodes fournies.")
  }
  
  # Transformation en format long
  results_melted <- melt(
    different_scenarios,
    id.vars = c("parameter", "scenarioB", "scenarioT"),
    measure.vars = methodnames_valid
  )
  names(results_melted) <- c("parameter", "ScenarioB", "ScenarioT", "Method", "ARI")
  
  # Résumé des données
  resdf <- data_summary(results_melted, "ARI", c("parameter", "Method", "ScenarioB", "ScenarioT"))
  resdf <- subset(resdf, parameter %in% xbreaks)
  # Adaptation automatique des couleurs, formes et linetypes
  n_methods <- length(methodnames_valid)
  colors <- c(
    "#000000", # noir
    "#E69F00", # orange
    "#56B4E9", # bleu ciel
    "#009E73", # vert
    "#F0E442", # jaune
    "#0072B2", # bleu
    "#D55E00", # vermillon
    "#CC79A7", # violet
    "#999999", # gris
    "#A65628"  # brun
  )[1:n_methods]          # Palette adaptée daltoniens
  shapes <- c(
    16, # cercle plein
    17, # triangle plein
    15, # carré plein
    18, # diamant plein
    3,  # plus
    4,  # croix
    8,  # étoile
    0,  # carré vide
    1,  # cercle vide
    2   # triangle vide
  )[1:n_methods]     # Formes adaptées
  linetypes <- c(
    "solid",
    "dashed",
    "dotted",
    "dotdash",
    "longdash",
    "twodash",
    "solid",
    "dashed",
    "dotted",
    "dotdash"
  )[1:n_methods]                            # Types de ligne
  
  # Création du plot
  p <- ggplot(resdf, aes(x = parameter, y = ARI)) +
    geom_line(aes(color = Method, linetype = Method)) +
    geom_point(aes(color = Method, shape = Method)) +
    # geom_errorbar(aes(ymin=ARI-2*se, ymax=ARI+2*se, color = Method), width=0.2) +
    ylim(ylim) +
    scale_x_continuous(breaks = xbreaks) +
    ylab(metric) +
    xlab(parameter_name) +
    theme_bw() +
    facet_grid(ScenarioT ~ ScenarioB) +
    scale_color_manual(labels = methodnames_valid, values = colors) +
    scale_shape_manual(labels = methodnames_valid, values = shapes) +
    scale_linetype_manual(labels = methodnames_valid, values = linetypes) +
    theme(legend.position = "top", legend.text.align = 0)
  
  return(p)
}

make_ggplot_single <- function(different_scenarios, parameter_name,
                               methodnames,
                               xbreaks = NULL,
                               metric="ARI",
                               ylim = c(0,1)) {

  require(ggplot2)
  require(reshape2)
  require(scales)

  # garder uniquement les méthodes demandées
  method_cols <- intersect(
    methodnames,
    names(different_scenarios)
  )

  if(length(method_cols) == 0){
    stop("Aucune méthode trouvée dans le tableau")
  }


  # format long
  results_melted <- melt(
    different_scenarios,
    id.vars = "parameter",
    measure.vars = method_cols,
    variable.name = "Method",
    value.name = "ARI"
  )


  # moyenne ARI pour chaque parameter et méthode
  resdf <- aggregate(
    ARI ~ parameter + Method,
    data = results_melted,
    FUN = mean,
    na.rm = TRUE
  )


  # ordre des méthodes comme demandé
  resdf$Method <- factor(
    resdf$Method,
    levels = method_cols
  )


  ggplot(
    resdf,
    aes(x = parameter,
        y = ARI,
        group = Method)
  ) +

    geom_line(
      aes(color = Method,
          linetype = Method)
    ) +

    geom_point(
      aes(color = Method,
          shape = Method)
    ) +

    ylim(ylim) +

    scale_x_log10(breaks = xbreaks) +

    ylab(metric) +
    xlab(parameter_name) +

    theme_bw() +

    scale_color_manual(
      values = hue_pal()(length(method_cols)),
      labels = method_cols
    ) +

    scale_shape_manual(
      values = rep(c(19,17,15,7,3,8,18,16,1,2,4),
                   length.out = length(method_cols))
    ) +

    scale_linetype_manual(
      values = rep(1:6,
                   length.out = length(method_cols))
    ) +

    theme(
      legend.position="top",
      legend.text.align=0
    )
}