# Degree-Corrected Joint Matrix Factorization for Multilayer Community Detection

Code for the paper **"Degree-Corrected Joint Matrix Factorization for Multilayer Community Detection"** by Alexandra Dache, Manon Rustin, Nicolas Gillis, and Arnaud Vandaele.

This repository contains:
- Python and MATLAB implementations of **mFROST**, our method for community detection in multilayer networks.
- Code to reproduce the numerical experiments presented in the paper, including synthetic experiments and experiments on real-world multilayer networks.

The synthetic experiments follow the settings of Agterberg, Lubberts, and Arroyo, *"Joint spectral clustering in multilayer degree-corrected stochastic blockmodels"*, and build on their [DC-MASE repository](https://github.com/jesusdaniel/dcmase).

The repository is organized into three main folders:

- `Python/`: the `mfrost` Python package and an example script.
- `Matlab/`: the MATLAB implementation of mFROST and an example script.
- `Experiments/`: all code and results for the numerical experiments conducted in R. 

## mFROST for community detection 

Given $L$ symmetric, nonnegative adjacency matrices $A_1,\ldots,A_L$ representing networks on the same $n$ nodes, and a prescribed number of communities $r$, mFROST jointly approximates each layer as $A_l \approx Z_l S_l Z_l^T$ by solving

$$
\min_{V,\{D_l,Z_l,S_l\}_{l=1}^L}
\sum_{l=1}^L \|A_l-Z_lS_lZ_l^T\|_F^2
$$

subject to

$$
Z_l=D_lV,\qquad Z_l^TZ_l=I_r,\qquad S_l\geq 0,
\qquad l=1,\ldots,L,
$$

where $V\in\{0,1\}^{n\times r}$ has exactly one nonzero entry per row and each $D_l$ is a diagonal matrix with nonnegative entries.

Where:
- **$A_l\in\mathbb{R}_+^{n\times n}$** is the adjacency matrix of layer $l$.
- **Z_l** is a $n \times r$ matrix encoding the assignment of each node into **r** communities, where $Z_l(i,k) \neq 0$ if node $i$ belongs to community $k$.
- The constraint $Z_l=D_l V imposes that the community assignments are the same across layers but the values of Z_l can change between layers to handle degree heterogeneity across layers (cf paper) 
- **S_l** is a $r\times r$ central matrix describing interactions between communities in the layer $l$.


### Python implementation 
The Python package is located in `Python/mfrost/`. An example demonstrating how to run mFROST is provided in `Python/Example.py`.

We recommend Python 3.12 or later. Install the dependencies using the provided `requirements.txt` file. From the directory containing that file, run:

```bash
python -m pip install -r requirements.txt
```

The Python implementation uses:
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`

`matplotlib` is also needed to generate the figures in `Example.py`.

### Matlab implementation 
The MATLAB implementation is provided in Matlab/, with all functions required by mFROST located in the algo/ subfolder. The main function is algo/mfrost/mfrost.m. Run Instal.m to add the required folders to the MATLAB path, then run Example.m for an example of how to use the algorithm.


## Reproduce the experiments
The synthetic experiments use the settings of Agterberg, Lubberts, and Arroyo and build on their [DC-MASE code](https://github.com/jesusdaniel/dcmase). We also include experiments on real-world multilayer networks.


#### Method
- *Experiments/Python/mfrost: python package for the mFROST method 
- *Experiments/Python/graphtool-script.py*: wrapper for running graphtool method.
- *Experiments/R/dcmase.R*: implements the degree-corrected adjacency spectral embedding.
- *Experiments/R/comdet-dcmase.R*: implements a community detection method based on DC-MASE.
- *Experiments/R/comdetmethods.R*: code implementation of alternative methods for multilayer community detection (including method by Paul and Chen (2020) contained in *R/Codes_Spectral_Matrix_Paul_Chen_AOS_2020.r*, and the method from the Python package graphtool, which is called via the script *R/run_graph_tool.R*)
- *Experiments/R/SpectralMethods.R* and *R/getElbows.R*: contain auxiliary functions to perform spectral embeddings.
- *Experiments/R/make_ggplot.R* and *R/make_dcsbm_plots.R*: implement functions to create plots from experiments.

#### Experiments
- *Experiments/Simulations/run_all_methods.R*: wrapper functions for generating simulated data and running all community detection methods on these simulations
- *Experiments/Simulations/simulations sparsity.R*: wrapper for data generation in simulations as function of network sparsity 


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
#### Python

Python must also be installed, together with the dependencies listed in the **Python implementation** section, to run mFROST from R. Configure `reticulate` to use the Python environment containing these dependencies before running the experiments:

```r
library(reticulate)
use_python("/path/to/python", required = TRUE)
```

Replace `/path/to/python` with the path to the appropriate Python executable.

Running the graph-tool baseline additionally requires `graph-tool` in the Python environment used by its wrapper.

## References
 TO DO : add our paper 
 
if you reproduce the synthetics experiments, please cite :
Agterberg, J., Lubberts, Z., & Arroyo, J. (2025). Joint spectral clustering in multilayer degree-corrected stochastic blockmodels. Journal of the American Statistical Association, (just-accepted), 1-23. 
[![arXiv shield](https://img.shields.io/badge/arXiv-2212.05053-red.svg?style=flat)](https://arxiv.org/abs/2212.05053)
