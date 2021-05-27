import time
import numpy as np
import utils

CONVERGENCE_LIMIT = 0.000001
GROUPING_DISTANCE = 0.00001

# Class that implements naive mean-shift algorithm
# pern: http://proceedings.mlr.press/v2/wang07d/wang07d.pdf
class MeanShift:
    # __init__(): Constructor for MeanShift class, takes in the kernel for the algorithm
    def __init__(self, kernel):
        if kernel == 'gaussian':
            self.kernel = utils.gaussian_kernel_ms
        if kernel == 'multivariate gaussian':
            self.kernel = utils.gaussian_kernel_correlations

    # do(): Perform the mean shift algorithm to find the modes
    # data is of the form: [ features ]
    #                      [ features ] <- records
    def do(self, reference_set, bandwidth, convergence_limit):
        # in this case, query set == reference set
        query_set = np.array(reference_set)

        # distance vector
        distances = [float('inf')] * len(reference_set)

        while max(distances) > convergence_limit:
            for point in range(len(query_set)):
                # save old point
                old_point = np.array(query_set[point])

                # reassign point
                query_set[point] = self.__compute_mean_shift(query_set[point], reference_set, bandwidth)

                # compute distance
                distances[point] = np.linalg.norm(query_set[point] - old_point)

        return MeanShiftResult(reference_set, query_set)

    # __compute_mean_shift(): Computes mean shift vector m(x) - x 
    # per https://en.wikipedia.org/wiki/Mean_shift
    def __compute_mean_shift(self, x_q, reference_set, bandwidth):
        # create weights array
        weights = np.array([self.kernel((np.linalg.norm(x_q - x_r) ** 2) / (bandwidth ** 2), bandwidth) for x_r in reference_set])

        # matmul weights by data
        numerator = np.matmul(reference_set.T, weights)
        return numerator / weights.sum()

# Class that returns result from a mean-shift algorithm
class MeanShiftResult:
    def __init__(self, reference_set, query_set):
        self.reference_set = reference_set
        self.query_set = query_set
