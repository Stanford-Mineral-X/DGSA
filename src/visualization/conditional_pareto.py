"""
Make Pareto plots showing single-parameter sensitivity.
Based upon the work of Celine Scheidt and Jihoon Park.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings

def conditional_pareto_by_bin(
        single_sensitivity_results: dict,
        conditional_sensitivity_results: dict,
        conditioning_parameter_name: str, 
        parameter_names: list[str],
        title: str = None,
        fig_size: tuple = (10, 6),
        font_size: int = 12,
        font: str = None
    ) -> None:
    """
    Make a Pareto plot showing the standardized measure of sensitivity values for each parameter and each cluster.

    Parameters
    ----------
    single_sensitivity_results : dict
        Results from single-parameter sensitivity analysis containing:
        'single_l1norm' or 'single_ASL'
            - 'by_cluster' : np.ndarray of shape (n_parameters, n_clusters)
                Sensitivity of each parameter across clusters.
            - 'standardized' : np.ndarray of shape (n_parameters,)
                Standardized sensitivity values for each parameter.
            - 'hypothesis_test' : np.ndarray of shape (n_parameters,), dtype=bool
                Boolean array indicating statistically significant sensitivities.
            _ 'alpha' : float
                A user-specified standard to determine if a parameter is sensitive or not
                For the l1norm method, it is the quantile of the bootstrapped distances.
                For the ASL method, it is used to perform a hypothesis test
            - 'sensitivity_method' : str
                Method to compute sensitivity: l1norm or ASL .

    conditional_sensitivity_results : dict
        Results from conditional parameter sensitivity analysis containing:
        'conditional_l1norm' or 'conditional_ASL'
            - 'by_cluster_and_bin' : np.ndarray of shape (n_parameters,n_parameters,n_clusters,n_bins)
                Conditonal sensitivity (1st index for conditioned parameter, 2nd for conditioning parameter, i.e., 1st|2nd) for each cluster and each bin
            - 'standardized' : np.ndarray of shape (n_parameters,n_parameters)
                Standardized condtional sensitivity over clusters and bins.
            - 'hypothesis_test' : np.ndarray of shape (n_parameters,n_parameters), dtype=bool
                Boolean array indicating statistically significant sensitivities.
            _ 'alpha' : float
                A user-specified standard to determine if a parameter is sensitive or not
                For the l1norm method, it is the quantile of the bootstrapped distances.
                For the ASL method, it is used to perform a hypothesis test
            - 'sensitivity_method' : str
                Method to compute sensitivity: l1norm or ASL.
    
    conditioning_parameter_name : str
        Name of conditioning parameter (e.g. y in x|y)
                
    parameter_names : list[str] 
        List of parameter names

    title : str, default = None
        Title for the plot. If None, no title is displayed.

    fig_size : tuple, default = (10,6)
        Figure size in inches (width, height)

    font_size : int, default = 12
        Font size for text in the plot

    font : str, default = None
        Font family to use (e.g. 'DejaVu Sans', 'Helvetica', 'Times New Roman').
        If None, matplotlib default is used.

    Returns
    -------
    A Pareto plot showing the standardized measure of sensitivity values for each parameter and each cluster..
    """
    
    # get dimensions
    n_params, _, _, n_bins = conditional_sensitivity_results['by_cluster_and_bin'].shape

    # convert inputs to numpy arrays
    parameter_names = np.asarray(parameter_names)

    # check if missing data
    required_keys = ['by_cluster', 'standardized', 'hypothesis_test', 'alpha', 'sensitivity_method']
    results_keys = set(single_sensitivity_results.keys())
    missing_keys = [key for key in required_keys if key not in results_keys]
    if missing_keys:
        # Use join to create a comma-separated string of the missing keys
        missing_str = ', '.join(missing_keys) 
        raise ValueError(f"The sensitivity results dictionary does not contain the required key(s): {missing_str}.")

    # check if parameter names are complete
    if parameter_names.size != single_sensitivity_results['standardized'].size:
        raise ValueError("parameter_names must match the sensitivity array length.")

    # check the names of variables
    try:
        idx_conditioning_parameter = np.where(parameter_names == conditioning_parameter_name)[0][0]
    except ValueError:
        raise ValueError(f'Enter correct name of conditioning parameters: {conditioning_parameter_name}')

    # calculate sensitivities per bin
    conditional_by_cluster_and_bin = conditional_sensitivity_results['by_cluster_and_bin'].copy()
    with warnings.catch_warnings(): # Suppress the NaN warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # sensitivity_by_bin = np.nanmax(conditional_by_cluster_and_bin, axis=2) # array shape (n_parameter,n_parameter,n_bins)
        sensitivity_by_bin = np.nanmean(conditional_by_cluster_and_bin, axis=2) # array shape (n_parameter,n_parameter,n_bins)
    sensitivity = sensitivity_by_bin[:, idx_conditioning_parameter, : ] # array shape (n_parameter,n_bins)

    # remove the conditioning parameter
    sensitivity = np.delete(sensitivity, idx_conditioning_parameter, axis=0)
    single_standardized = single_sensitivity_results['standardized'].copy()
    single_standardized = np.delete(single_standardized, idx_conditioning_parameter, axis=0)
    parameter_names = np.delete(parameter_names, idx_conditioning_parameter, axis=0)
    n_params -= 1

    # sort from least sensitive to most sensitive
    sort_idx = np.argsort(single_standardized)
    sorted_sensitivity = sensitivity[sort_idx]
    sorted_names = parameter_names[sort_idx]

    # choose font
    if font is not None:
        plt.rcParams['font.family'] = font

    # create figure and axis
    fig, ax = plt.subplots(figsize = fig_size)
    ax.set_axisbelow(True)  # Put grid behind bars
    y_pos = np.arange(n_params)
    
    # set up bar height and colors
    total_bar_height = 0.8
    bar_height = total_bar_height / n_bins
    colors = plt.cm.tab10(np.linspace(0, 1, n_bins))
    # colors = plt.cm.jet(np.linspace(0, 1, n_bins))

    # make the plot
    if single_sensitivity_results['sensitivity_method'] == 'l1norm':
        sensitivity_to_plot = sorted_sensitivity

    elif single_sensitivity_results['sensitivity_method'] == 'ASL':
        # Transform sensitivities to z-scores for visualization
        # sorted_sensitivity_zscore = norm.ppf(sorted_sensitivity, loc=2, scale=1)
        sorted_sensitivity_zscore = norm.ppf(sorted_sensitivity/100, loc=3, scale=1)
        # Replace NaN/inf with max + eps
        finite_mask = np.isfinite(sorted_sensitivity_zscore)
        if not np.all(finite_mask):
            max_val = np.max(sorted_sensitivity_zscore[finite_mask])
            sorted_sensitivity_zscore[~finite_mask] = max_val + np.finfo(float).eps
        sensitivity_to_plot = sorted_sensitivity_zscore

    legend_labels = ['Low Bin', 'Medium Bin', 'High Bin']
    for c in range(n_bins):
        offset = (c - (n_bins - 1) / 2) * bar_height
        ax.barh(y_pos + offset, sensitivity_to_plot[:, c], height=bar_height, color=colors[c], label=legend_labels[c])

    # customize figure
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.set_xlabel(f'Sensitivity Measure for Parameter on y-axis|{conditioning_parameter_name}', fontsize=font_size)
    ax.set_title(title, fontsize=font_size+2)
    ax.legend(fontsize=font_size)

    if single_sensitivity_results['sensitivity_method']  == 'ASL':
        # x tick labels
        # TickToDisplay = np.array([0.05, 0.2, 0.5, 0.8, 0.95, 0.993, 0.9995, 0.99999])
        TickToDisplay = np.array([5, 20, 50, 80, 95, 99.3, 99.95, 99.999])
        ax.set_xticks(norm.ppf(TickToDisplay/100, 2, 1))
        ax.set_xticklabels(TickToDisplay)

    plt.show()


