import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-5.0, +0.0],
                        [+0.0, +5.0],
                        [+5.0, +0.0],
                        [+0.0, -5.0]])
group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
    
    distances = dt.cdist(X, centroids, metric="euclidean")
    assignments = np.argmin(distances, axis=1)
    
    means = []
    covariances = []
    priors = []
    for k in range(K):
        cluster_points = X[assignments == k]
        means.append(cluster_points.mean(axis=0))
        covariances.append(np.cov(cluster_points, rowvar=False))
        priors.append(cluster_points.shape[0] / X.shape[0])
    
    means = np.array(means)
    covariances = np.array(covariances)
    priors = np.array(priors)
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    num_samples, num_features = X.shape
    responsibilities = np.zeros((num_samples, K))
    max_iterations = 100

    for iteration in range(max_iterations):
        for k in range(K):
            responsibilities[:, k] = priors[k] * stats.multivariate_normal.pdf(
                X, mean=means[k], cov=covariances[k]
            )
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-step: Update parameters
        for k in range(K):
            N_k = responsibilities[:, k].sum()
            means[k] = (X * responsibilities[:, k][:, np.newaxis]).sum(axis=0) / N_k
            covariances[k] = np.cov(
                X.T,
                aweights=responsibilities[:, k],
                ddof=0,
            )
            priors[k] = N_k / num_samples

    assignments = np.argmax(responsibilities, axis=1)
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below

    plt.figure(figsize=(10, 10))
    
    # Use colormap and slice it for K colors
    cmap = plt.cm.get_cmap("tab10")  # Get the full colormap
    colors = [cmap(i) for i in range(K)]  # Extract K distinct colors
    
    # Plot data points with borders
    for k in range(K):
        cluster_points = X[assignments == k]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {k+1}",
            color=colors[k],    # Cluster color
            edgecolors="black", # Border color
            linewidth=0.6,      # Border thickness
            s=20                # Marker size
        )
    
    # Plot original Gaussian densities (dashed lines)
    for mean, cov in zip(group_means, group_covariances):
        x, y = np.meshgrid(
            np.linspace(-8, 8, 100), np.linspace(-8, 8, 100)
        )
        pos = np.dstack((x, y))
        plt.contour(
            x,
            y,
            stats.multivariate_normal(mean, cov).pdf(pos),
            levels=[0.01],
            colors="black",
            linestyles="dashed",
        )
    
    # Plot final Gaussian densities (solid lines)
    for k, (mean, cov) in enumerate(zip(means, covariances)):
        x, y = np.meshgrid(
            np.linspace(-8, 8, 100), np.linspace(-8, 8, 100)
        )
        pos = np.dstack((x, y))
        plt.contour(
            x,
            y,
            stats.multivariate_normal(mean, cov).pdf(pos),
            levels=[0.01],
            colors=[colors[k]],  # Ensure proper color is used
        )
    
    # Finalize the plot
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("EM Clustering Results with Cluster Borders and Gaussian Contours")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)
