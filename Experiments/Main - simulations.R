#######################################
# Main simulation results from "Joint Spectral Clustering in
# Multilayer Degree Corrected Blockmodles"
#######################################

#NOTE: To run mFROST and graph-tool Python must also be installed, together with the dependencies listed in the requirements.txt, 
# Configure reticulate to use the Python environment containing these dependencies before running the experiments:
# Modify the file Experiments/R/comdetmethods to use mFROST in Python

# Load all methods for simulations
source("Simulations/run_all_methods.R")
library(dplyr)
# Note: 

######################################
#Simulation settings
######################################

num_nodes <- 150
K <- 3
degree_distribution <- "exp" # "exp" for exponnential or "pow" for power distribution 
num_replications <- 100
num_layers <- list(1,2,3,5,7,10,15,20,30,40,50)


parameters_list <- num_layers
param_iter = parameters_list
parameters_list <- lapply(num_layers, function(x) {
  list(
    m = x,
    n = num_nodes,
    K = K,
    degree_distribution = degree_distribution
  )
})


#######################################
# Run different scenarios
#######################################

# Same B same theta

results_simulation1 <- iterate_parameters(sim_setting = simulation1, parameters_list, param_iter, num_replications)
save(results_simulation1,file="sim1.RData")
resume <- results_simulation1 %>%
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
  file = "simulation1.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


# Different B same theta
results_simulation2 <- iterate_parameters(sim_setting = simulation2, parameters_list, param_iter, num_replications)
save(results_simulation2,file="sim2.RData")
resume <- results_simulation2 %>%
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
  file = "simulation2.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


# Diff B diff theta
results_simulation3 <- iterate_parameters(simulation3, parameters_list, param_iter, num_replications)
save(results_simulation3,file="sim3.RData")
resume <- results_simulation3 %>%
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
  file = "simulation3.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
# Same B different theta
results_simulation4 <- iterate_parameters(simulation4, parameters_list, param_iter, num_replications)
save(results_simulation4,file="sim4.RData")
resume <- results_simulation4 %>%
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
  file = "simulation4.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)



# Same B alternating theta
results_simulation6 <- iterate_parameters(simulation6, parameters_list, param_iter, num_replications)
save(results_simulation6,file="sim6.RData")
resume <- results_simulation6 %>%
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
  file = "simulation6.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

# Different B alternating theta
results_simulation7 <- iterate_parameters(simulation7, parameters_list, param_iter, num_replications)
save(results_simulation7,file="sim7.RData")
resume <- results_simulation7 %>%
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
  file = "simulation7.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


#######################################
# Combine results
#######################################
results_simulation1$scenarioB <- "Same θ"
results_simulation2$scenarioB <- "Different θ"
results_simulation3$scenarioB <- "Different θ"
results_simulation4$scenarioB <- "Same θ"
results_simulation6$scenarioB <- "Same θ"
results_simulation7$scenarioB <- "Different θ"

results_simulation1$scenarioT <- "Same D"
results_simulation2$scenarioT <- "Same D"
results_simulation3$scenarioT <- "Different D"
results_simulation4$scenarioT <- "Different D"
results_simulation6$scenarioT <- "Alternating D"
results_simulation7$scenarioT <- "Alternating D"

different_scenarios <- rbind(results_simulation1, results_simulation2, results_simulation4, results_simulation3, 
                             results_simulation6, results_simulation7)

different_scenarios$scenarioB <- factor(different_scenarios$scenarioB,
                                       levels = c("Same θ", "Different θ"))
different_scenarios$scenarioT <- factor(different_scenarios$scenarioT,
                                        levels = c("Same D", "Different D", "Alternating D"))
save(different_scenarios, file = "Results-testcomplete.RData")

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


p <- make_ggplot_multipleBT2(different_scenarios_metric, "Number of layers", xbreaks = c(1,10,20,30,40,50), metric,methodnames = c("mFROST","OLMF","DC-MASE","graph-tool","Sum A","Bias-adjusted SoS","MASE"),ylim = c(0,0.6))#, "graph-tool"))
ggsave(
  filename = "simulations.png",
  plot = p,              # ton objet ggplot
  width = 8,
  height = 6,
  units = "in",
  dpi = 600,
  bg = "white"
)
#make_ggplot_multipleBT2(different_scenarios, "Number of graphs", xbreaks = c(1, seq(10, 50, 10)),
                        #methodnames = c("DC-MASE", "Sum of adj. matrices", "Bias-adjusted SoS",  "MASE", "OLMF"))#, "graph-tool"))

#dev.off()


