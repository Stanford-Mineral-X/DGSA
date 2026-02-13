from numpy.typing import NDArray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

def cluster_MDS(
    distance_matrix: NDArray[np.float64], 
    clustering: dict,
    stress_score_output_dimensions: int,
    fig_size: tuple = (8, 6),
    font_size: int = 12,
    font: str = None
    ) -> tuple[plt.Figure, plt.Figure]:
    """
    Perform multidimensional scaling (MDS) on the distance matrix to plot (1) clusters in 2D space and (2) stress scores across dimensions.

    Parameters
    ----------
    distance_matrix : np.ndarray of shape (n_samples, n_samples)
        Array of distances between samples 
          
    clustering : dict
        Results from clustering analysis containing:
          - cluster_assignments : np.ndarray of shape (n_samples,)
              Cluster index for each sample.
          - medoid_indices : np.ndarray of shape (n_clusters,)
              Indices of cluster medoids.
          - n_points : np.ndarray of shape (n_clusters,)
              Number of points belonging to each cluster.

    stress_score_output_dimensions : int
        Maximum number of dimensions to compute stress scores for.
    
    fig_size : tuple, default = (10,6)
        Figure size in inches (width, height)

    font_size : int, default = 12
        Font size for text in the plot

    font : str, default = None
        Font family to use (e.g. 'DejaVu Sans', 'Helvetica', 'Times New Roman').
        If None, matplotlib default is used.
              
    Returns:
    -------
    tuple[plt.Figure, plt.Figure]
        A tuple containing two figures:
        (1) MDS plot of clusters in 2D space
        (2) Stress scores across dimensions
    """
    dimensions = list(range(1, stress_score_output_dimensions + 1))
    stress_scores = []
    mds_coordinates_2d = None # Will store the 2D MDS coordinates for scatter plot
    
    # run MDS
    for dim in dimensions:
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42)
        mds_coordinates = mds.fit_transform(distance_matrix)
        if dim == 2:
            mds_coordinates_2d = mds_coordinates

        # calculate stress score (Kruskal's Stress-1 formula)
        dist_low_dim = pairwise_distances(mds_coordinates)
        stress = np.sqrt(np.sum((distance_matrix - dist_low_dim) ** 2) /
                         np.sum(distance_matrix ** 2))
        stress_scores.append(stress)
    
    # prepare DataFrame for plotting
    df = pd.DataFrame(mds_coordinates_2d, columns=['Dimension 1', 'Dimension 2'])
    df['cluster'] = clustering['cluster_assignments']

    # define colormap
    unique_clusters = np.sort(df['cluster'].unique())
    cmap = plt.get_cmap('tab10', len(unique_clusters))  # Safe for <=10 clusters

    # choose font
    if font is not None:
        plt.rcParams['font.family'] = font

    # plot MDS for each cluster
    fig_mds, ax_mds = plt.subplots(figsize = fig_size, constrained_layout=True)
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        ax_mds.scatter(
            cluster_data['Dimension 1'],
            cluster_data['Dimension 2'],
            color=cmap(i),
            label=f'Cluster {cluster_id + 1}',
            s=60,  # size of points
            edgecolor='k',  # optional: black edge for visibility
            alpha=0.8
        )

    ax_mds.set_xlabel('Dimension 1', fontsize=font_size)
    ax_mds.set_ylabel('Dimension 2', fontsize=font_size)
    ax_mds.set_title('MDS of Response Clusters', fontsize=font_size + 2)
    # ax_mds.legend(title="Clusters", loc='upper right')
    ax_mds.legend(fontsize=font_size - 2)
    ax_mds.grid(True, linestyle='--', alpha=0.6)

    # plot stress scores
    fig_stress_score, ax_stress_score = plt.subplots(figsize = fig_size, constrained_layout=True)
    ax_stress_score.plot(dimensions, stress_scores, marker='o', linestyle='-', color='b')
    ax_stress_score.set_xlabel('Number of Dimensions', fontsize=font_size)
    ax_stress_score.set_ylabel('Stress Score', fontsize=font_size)
    ax_stress_score.set_title('Stress Scores for Different Dimensions', fontsize=font_size + 2)
    ax_stress_score.grid(True, linestyle='--', alpha=0.6)
    
    return fig_mds, fig_stress_score

