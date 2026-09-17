# Code for "Degree-Corrected Joint Matrix Factorization for
Multilayer Community Detection" by Alexandra Dache, Manon Rustin, Nicolas Gillis and Arnaud Vandaele 

This repository contains 
- mFROST, our method for community detection in multilayer networks implemented in Python and in Matlab 
- all the code required to reproduce the experiments from the paper "Degree-Corrected Joint Matrix Factorization for
Multilayer Community Detection" based on the paper "Joint spectral clustering in multilayer degree corrected blockmodels" by Joshua Agterberg, Zachary Lubberts and Jesus Arroyo from the github  jesusdaniel/dcmase  .

folder Python => package mfrost and example.py 
folder Matlab => folder mfrots and eaxample.py
folder Experiments


## mFROST for community detection 

Given a list of  *L*  adjacency matrices representing graphs defined over the same nodes (cf multilayer networks) and the number of communities *r* we are looking for,

mFROST aims to solve the following joint optimization problem:
$$
\min_{D_l \geq 0,V\geq 0, S \geq 0} \sum_{l=1}^L||A_l - Z_lS_lZ_l^T||_F^2 \quad \text{s.t.} \quad Z_l^TZ_l = I, Z_l=D_lV
$$

Where:
- **A_l** is a given symmetric nonnegative matrix of size $n \times n$  (e.g., the adjacency matrix of the layer $l).
- **Z_l** is a $n \times r$ matrix encoding the assignment of each node into **r** communities, where $Z(i,k) \neq 0$ if node $i$ belongs to community $k$.
- The constraint $Z_l=D_V imposes that the community assignments are the same across layers but the values of Z_l can change between layers to handle degree heterogeneity across layers (cf paper) 
- **S_l** is a $r\times r$ central matrix describing interactions between communities in the layer $l$.

The algorithm is implemented in Python and matlab 

### Python implementation 
folder Python
An example script demonstrating how to use the `FROST` function is included in the script`Example.py`.


package mfrost inthe folder python\mfrost
to use it we recommend python > 3.12
To install the dependencies requirements.txt file 
mFROST required the librairies below: 

numpy
scipy
pandas
scikit-learn
(matplotlib) for the Figures in Example.py  

### Matlab implementation 
folder algo/ mfrost/mfrost.m 
An example script demonstrating how to use the `FROST` function is included in the script`Example.m`.


## Reproduce the experiments
The experiments are based on the paper "Joint spectral clustering in multilayer degree corrected blockmodels" by Joshua Agterberg, Zachary Lubberts and Jesus Arroyo from the github  jesusdaniel/dcmase  .
We kepts the same settings for the synthetics experiments and added real-world multilayer networks. 


### File overview section
The folder contains scripts to implement the methods, implement the experiment, run the experiments and generate figures, and additional files containing the results of the experiments. Descriptions of each file are listed below.

#### Method
- *Python/mfrost: python package for the mFROST method 
- *Python/graphtool-script.py*: wrapper for running graphtool method.
- *R/dcmase.R*: implements the degree-corrected adjacency spectral embedding.
- *R/comdet-dcmase.R*: implements a community detection method based on DC-MASE.
- *R/comdetmethods.R*: code implementation of alternative methods for multilayer community detection (including method by Paul and Chen (2020) contained in *R/Codes_Spectral_Matrix_Paul_Chen_AOS_2020.r*, and the method from the Python package graphtool, which is called via the script *R/run_graph_tool.R*)
- *R/SpectralMethods.R* and *R/getElbows.R*: contain auxiliary functions to perform spectral embeddings.
- *R/make_ggplot.R* and *R/make_dcsbm_plots.R*: implement functions to create plots from experiments.

#### Experiments
- *Experiments/run_all_methods.R*: wrapper functions for generating simulated data and running all community detection methods on these simulations
- *Experiments/simulations sparsity.R*: wrapper for data generation in simulations as function of network sparsity 


#### Code to generate figures
- *Main - simulations.R*: Run simulation experiments for Figure 2 of the paper. 
- *Supplement - Simulations sparsity.R*: Run simulation experiments for Figure 3 of the paper. 


#### Requirements

To run this project successfully, ensure you have the following:

##### R Version
- R (>= 4.2.1)

##### Required R Packages
Several packages are required to run the code and generate the figures. These are listed below, including the version of each package that was used. To install these versions, run the following code.
The code requires the package listed below, and was run using the versions

```r
install.packages("remotes") # If not already installed
remotes::install_version("igraph", version = "1.4.2")
remotes::install_version("mclust", version = "5.4.7")
remotes::install_version("ScorePlus", version = "0.1")
remotes::install_version("reshape2", version = "1.4.4")
remotes::install_version("ggplot2", version = "3.3.5")
remotes::install_version("grid", version = "4.1.1")
remotes::install_version("plyr", version = "1.8.6")
remotes::install_version("ggthemes", version = "4.2.4")
remotes::install_version("maps", version = "3.4.0")
remotes::install_version("mapdata", version = "2.3.0")
remotes::install_version("parallel", version = "4.1.1")

# Interface between R and Python !!! 
install.packages("reticulate")
```
Python must also be installed, along with the libraries listed in the Python implementation section if you want to use mFROST.

# References
 TO DO : add our paper 
if you reproduce the synthetics experiments, please cite :
Agterberg, J., Lubberts, Z., & Arroyo, J. (2025). Joint spectral clustering in multilayer degree-corrected stochastic blockmodels. Journal of the American Statistical Association, (just-accepted), 1-23. 
[![arXiv shield](https://img.shields.io/badge/arXiv-2212.05053-red.svg?style=flat)](https://arxiv.org/abs/2212.05053)
