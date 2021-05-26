import numpy as np

# Probabiltity density functions (might not need)
# ----------------------------------------------

# Normal probability density function
def normal_pdf(x, stdev, mean):
    return np.exp(-0.5 * ((x - mean) / stdev) ** 2) / (stdev * np.sqrt(2 * np.pi))

# Multivariate normal probability density function (takes into account covariances between inputs)
def multivariate_normal_pdf(x, mean, cov_matrix):
    numerator = np.exp(-0.5 * np.linalg.multi_dot([(x - mean).T, np.linalg.inv(cov_matrix), (x - mean)]))
    denominator = np.sqrt(((2 * np.pi) ** len(x)) * np.linalg.det(cov_matrix))
    return numerator / denominator

# Kernel functions
# ----------------------------------------------

# Standard gaussian kernel for mean shift (works for dimension >= 1)
# per https://en.wikipedia.org/wiki/Mean_shift
def gaussian_kernel_ms(x, bandwidth):
    numerator = np.exp(-0.5 * ((x / bandwidth) ** 2))
    denominator = (bandwidth * np.sqrt(2 * np.pi))
    return numerator / denominator

# Gaussian kernel where correlations between features matter (hence use of bandwidth (variance-covariance) matrix)
# per https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation
def gaussian_kernel_correlations(x, bandwidth_matrix):
    numerator = np.exp(-0.5 * (np.linalg.multi_dot([x.T, np.linalg.inv(bandwidth_matrix), x])))
    denominator = (np.sqrt(2 * np.pi) ** len(x)) * np.sqrt(np.linalg.det(bandwidth_matrix))
    return numerator / denominator

# Standard gaussian kernel for kde
# per https://www.youtube.com/watch?v=ZNwZkxpCGHE
def gaussian_kernel_kde(u):
    denominator = np.sqrt(2 * np.pi) ** u.shape[0]  
    numerator = np.exp(-0.5 * (np.linalg.norm(u) ** 2))
    return numerator / denominator
