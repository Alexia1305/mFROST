#######################################
# Supplementary simulation results from "Joint Spectral Clustering in
# Multilayer Degree Corrected Blockmodles"
#######################################

# These simulations compare the performance
# of different multilayer clustering methods
# in terms of edge density

source("Simulations/run_all_methods.R")
source("Simulations/simulations sparsity.R")
library(Matrix)
library(igraph)
library(mclust)

library(dplyr)

num_replications <- 100
ave_degs <- seq(2, 24, 2)


# Same B same theta
parameters_list <- ave_degs
param_iter = ave_degs/150
results_simulation1s <- iterate_parameters(sim_setting = simulation1s, parameters_list, param_iter, num_replications)
save(results_simulation1s,file="sim1s.RData")
resume <- results_simulation1s %>%
  dplyr::group_by(parameter, Method) %>%
  dplyr::summarise(
    dplyr::across(
      c(Error, NMI, ARI),
      list(
        moyenne = ~mean(.x, na.rm = TRUE),
        ecart_type = ~sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

resume[-c(1,2)] <- lapply(resume[-c(1,2)], round, digits = 4)

write.table(
  resume,
  file = "simulation1s.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

# Diif B same theta

results_simulation2s <- iterate_parameters(sim_setting = simulation2s, parameters_list, param_iter, num_replications)
save(results_simulation2s,file="sim2s.RData")
resume <- results_simulation2s %>%
  dplyr::group_by(parameter, Method) %>%
  dplyr::summarise(
    dplyr::across(
      c(Error, NMI, ARI),
      list(
        moyenne = ~mean(.x, na.rm = TRUE),
        ecart_type = ~sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

resume[-c(1,2)] <- lapply(resume[-c(1,2)], round, digits = 4)

write.table(
  resume,
  file = "simulation2s.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)



# Diff B diff theta

results_simulation3s <- iterate_parameters(simulation3s, parameters_list, param_iter, num_replications)
save(results_simulation3s,file="sim3s.RData")
resume <- results_simulation3s %>%
  dplyr::group_by(parameter, Method) %>%
  dplyr::summarise(
    dplyr::across(
      c(Error, NMI, ARI),
      list(
        moyenne = ~mean(.x, na.rm = TRUE),
        ecart_type = ~sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

resume[-c(1,2)] <- lapply(resume[-c(1,2)], round, digits = 4)

write.table(
  resume,
  file = "simulation3s.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)



# Same B different theta

results_simulation4s <- iterate_parameters(simulation4s, parameters_list, param_iter, num_replications)
save(results_simulation4s,file="sim4s.RData")
resume <- results_simulation4s %>%
  dplyr::group_by(parameter, Method) %>%
  dplyr::summarise(
    dplyr::across(
      c(Error, NMI, ARI),
      list(
        moyenne = ~mean(.x, na.rm = TRUE),
        ecart_type = ~sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

resume[-c(1,2)] <- lapply(resume[-c(1,2)], round, digits = 4)

write.table(
  resume,
  file = "simulation4s.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)




#############################################

# Alternating degrees
param_iter = ave_degs/150
parameters_list <- lapply(ave_degs, 
                          function(x) c(x, 150, 3))
results_simulation6s <- iterate_parameters(simulation6s, parameters_list, param_iter, num_replications)
save(results_simulation6s,file="sim6s.RData")
resume <- results_simulation6s %>%
  dplyr::group_by(parameter, Method) %>%
  dplyr::summarise(
    dplyr::across(
      c(Error, NMI, ARI),
      list(
        moyenne = ~mean(.x, na.rm = TRUE),
        ecart_type = ~sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

resume[-c(1,2)] <- lapply(resume[-c(1,2)], round, digits = 4)

write.table(
  resume,
  file = "simulation6s.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

results_simulation7s <- iterate_parameters(simulation7s, parameters_list, param_iter, num_replications)
save(results_simulation7s,file="sim7s.RData")
resume <- results_simulation7s %>%
  dplyr::group_by(parameter, Method) %>%
  dplyr::summarise(
    dplyr::across(
      c(Error, NMI, ARI),
      list(
        moyenne = ~mean(.x, na.rm = TRUE),
        ecart_type = ~sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

resume[-c(1,2)] <- lapply(resume[-c(1,2)], round, digits = 4)

write.table(
  resume,
  file = "simulation7s.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

#######################################
# Combine results
#######################################
results_simulation1s$scenarioB <- "Same θ"
results_simulation2s$scenarioB <- "Different θ"
results_simulation3s$scenarioB <- "Different θ"
results_simulation4s$scenarioB <- "Same θ"
results_simulation6s$scenarioB <- "Same θ"
results_simulation7s$scenarioB <- "Different θ"

results_simulation1s$scenarioT <- "Same D"
results_simulation2s$scenarioT <- "Same D"
results_simulation3s$scenarioT <- "Different D"
results_simulation4s$scenarioT <- "Different D"
results_simulation6s$scenarioT <- "Alternating D"
results_simulation7s$scenarioT <- "Alternating D"

different_scenarios <- rbind(results_simulation1s, results_simulation2s, results_simulation4s, results_simulation3s, 
                             results_simulation6s, results_simulation7s)

different_scenarios$scenarioB <- factor(different_scenarios$scenarioB,
                                       levels = c("Same θ", "Different θ"))
different_scenarios$scenarioT <- factor(different_scenarios$scenarioT,
                                        levels = c("Same D", "Different D", "Alternating D"))
save(different_scenarios, file = "Results-testcompletesparsity.RData")

source("R/make_ggplot.R")
#######################################
# Plot simulation results
#######################################
#load("Results-testcomplete.RData")

library(tidyr)
metric <- "Error"

different_scenarios_metric <- different_scenarios %>%
  pivot_wider(
    id_cols = c(Seed, parameter, scenarioB, scenarioT),
    names_from = Method,
    values_from = all_of(metric)
  )

png("Simulation-rep100-6scenarios-flipped.png", width = 1200, height = 1500, res = 200)


p <- make_ggplot_multipleBT2(different_scenarios_metric, "Edge density", xbreaks = c(0.01,0.04,0.07,0.10,0.13,0.16), metric,methodnames = c("mFROST","OLMF","DC-MASE","graph-tool","Sum A","Bias-adjusted SoS","MASE"),ylim = c(0,0.7))#, "graph-tool"))
ggsave(
  filename = "simulations.png",
  plot = p,              # ton objet ggplot
  width = 8,
  height = 6,
  units = "in",
  dpi = 600,
  bg = "white"
)
#dev.off()


