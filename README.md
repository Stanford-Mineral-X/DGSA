# Distance-based Generalized Sensitivity Analysis (DGSA)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
A Python package for performing distance-based generalized sensitivity analysis

## Authors
Jihui Ding, jihuid@stanford.edu

## Features

### Sensitivity Analysis
- **Single parameter sensitivity** computation  
- **Conditional (two-way) parameter sensitivity** to quantify asymmetric parameter interactions

### Visualization Tools
Generates a variety of plots using perceptually uniform colormaps:
- **Pareto plots** for sensitivity ranking
- **Heatmaps** for conditional interactions
- **CDF plots** for parameter distributions
- **MDS (Multidimensional Scaling)** for clustering visualizations

### Example Outputs
Below are example plots produced using the included sample dataset:

<img src="results/Park2016_single_l1norm.tiff" width="600" alt="Single parameter sensitivity plot">
<img src="results/Park2016_conditional_ASL.tiff" width="600" alt="Conditional parameter sensitivity plot">
<img src="results/Park2016_conditional_CDF.tiff" width="500" alt="Conditional CDF plot">
<img src="results/Park2016_MDS.tiff" width="500" alt="MDS cluster plot">

## Installation

### Requirements
- Python >= 3.10

### Required Packages
| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Plotting and visualization |
| `scipy` | Statistical fitting and bootstrap |
| `scikit-learn` | MDS for clustering visualization |
| `cmcrameri` | Perceptually uniform colormaps |

### Install
Clone the repository:
```bash
git clone https://github.com/<your-username>/DGSA.git
cd DGSA
```
Then install using one of the following:
```bash
# For regular use
pip install .

# For development (changes to source code take effect immediately)
pip install -e .
```

## How to Use

1. **Install the package**  
   Follow the Installation instructions above before running any notebooks.

2. **Explore the Example Notebooks**  
   Run the Jupyter notebooks included in this repository to learn the DGSA workflow (computation and visualization) using the provided example dataset.

3. **Prepare Your Own Data**  
   Format your inputs to match the example files:
   - **Parameter file:** a CSV containing model parameters.
   - **Response file:** either a CSV of responses or a **distance matrix** CSV.  
   Ensure the column structure and file format follow the examples.

4. **Run DGSA with Your Data**  
   Replace the example data paths in the notebooks with your own files and adjust any DGSA settings as needed.

## Repository Structure

```plaintext
repo_root
├── data
│   ├── example_responses.csv
│   ├── Park2016_distance_matrix.csv
│   └── Park2016_parameters.csv
├── notebooks
│   ├── DGSA_computation_Park2016.ipynb
│   └── DGSA_visualization_Park2016.ipynb
├── results
│   ├── Park2016_DGSA_results.pkl
│   ├── Park2016_MDS.tiff
│   ├── Park2016_conditional_ASL.tiff
│   ├── Park2016_conditional_CDF.tiff
│   ├── Park2016_conditional_l1norm.tiff
│   ├── Park2016_single_ASL.tiff
│   └── Park2016_single_l1norm.tiff
├── src
│   └── dgsa
│       ├── computation
│       │   ├── conditional_parameter_sensitivity.py
│       │   ├── kmedoids.py
│       │   └── single_parameter_sensitivity.py
│       ├── utils
│       │   └── dgsa_save_load.py
│       └── visualization
│           ├── cluster_mds.py
│           ├── conditional_cdf.py
│           ├── conditional_heatmap.py
│           ├── conditional_pareto.py
│           ├── single_cdf.py
│           └── single_pareto.py
├── tests
│   ├── fixtures
│   │   ├── Park2016_DGSA_results.pkl
│   │   ├── Park2016_distance_matrix.csv
│   │   └── Park2016_parameters.csv
│   ├── conftest.py
│   ├── test_clustering.py
│   ├── test_conditional_sensitivity.py
│   └── test_single_sensitivity.py
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```
## Acknowledgements

**Methodology developed by:**

Fenwick, D., Scheidt, C., & Caers, J. (2014). Quantifying asymmetric parameter interactions in sensitivity analysis: Application to reservoir modeling. *Mathematical Geosciences, 46(4), 493–511.*

**Example data from:**

Park, J., Yang, G., Satija, A., Scheidt, C. and Caers, J. (2016). DGSA: A Matlab toolbox for distance-based generalized sensitivity analysis of geoscientific computer experiments. *Computers & geosciences, 97, 15-29.*

**Previous implementations**

| Version | Author | Repository |
|--------|--------|------------|
| MATLAB | Céline Scheidt | https://github.com/lewisli/QUSS |
| MATLAB | Jihoon Park | https://github.com/SCRFpublic/DGSA |
| Python (light) | David Yin | https://github.com/SCRFpublic/DGSA_Light |

We acknowledge and appreciate the contributions of the above researchers and developers whose work laid the foundation for this implementation.



