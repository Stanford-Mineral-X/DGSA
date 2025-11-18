# Distance-based Generalized Sensitivity Analysis (DGSA)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Overview
A Python package for performing distance-based generalized sensitivity analysis
## Authors
Jihui Ding, jihuid@stanford.edu
## Features
- Compute single parameter sensitivity and condtional parameter sensitivity (two-way parameter interaction)
- Generate Pareto, heatmap, CDF plots using a perceptually uniform colormap

![single parameter sensitivity](example_plots/Park2016_single_l1norm.png)
![conditional parameter sensitivity](example_plots/Park2016_conditional_ASL.png)
![conditional CDF](example_plots/Park2016_conditional_CDF.png)
![MDS cluster](example_plots/Park2016_MDS.png)

## How to Use

1. **Explore the Example Notebooks**  
   Run the Jupyter notebooks included in this repository to learn the DGSA workflow (computation and visualization) using the provided example dataset.

2. **Prepare Your Own Data**  
   Format your inputs to match the example files:
   - **Parameter file:** a CSV containing model parameters.
   - **Response file:** either a CSV of responses or a **distance matrix** CSV.  
   Ensure the column structure and file format follow the examples.

3. **Run DGSA with Your Data**  
   Replace the example data paths in the notebooks with your own files and adjust any DGSA settings as needed.
