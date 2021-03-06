import numpy as np
import utils

from kdtree import KDTree

# Class that implements naive mean-shift algorithm
# per: http://proceedings.mlr.press/v2/wang07d/wang07d.pdf
class MeanShift:
    # __init__(): Constructor for MeanShift class, takes in the kernel for the algorithm
    # input: kernel to use in mean shift for kde
    def __init__(self, kernel, reference_set):
        if kernel == 'gaussian':
            self.kernel = utils.gaussian_kernel_ms
        if kernel == 'multivariate gaussian':
            self.kernel = utils.gaussian_kernel_correlations

        self.reference_set = reference_set

    # do(): Perform the mean shift algorithm to find the modes
    # input: query set, reference set, converging limit
    # output: converged query set
    def do(self, bandwidth, convergence_limit):
        # in this case, query set == reference set
        query_set = np.array(self.reference_set)

        # distance vector
        distances = [float('inf')] * len(self.reference_set)

        while max(distances) > convergence_limit:
            for point in range(len(query_set)):
                # save old point
                old_point = np.array(query_set[point])

                # reassign point
                query_set[point] = self.__compute_mean_shift(query_set[point], self.reference_set, bandwidth)

                # compute distance
                distances[point] = np.linalg.norm(query_set[point] - old_point)

        return MeanShiftResult(self.reference_set, query_set)

    # __compute_mean_shift(): Computes mean shift vector m(x) - x 
    # per https://en.wikipedia.org/wiki/Mean_shift
    # input: point in query set, reference set
    # output: shifted point in query set
    def __compute_mean_shift(self, x_q, reference_set, bandwidth):
        # create weights array
        weights = np.array([self.kernel((np.linalg.norm(x_q - x_r) ** 2) / (bandwidth ** 2), bandwidth) for x_r in reference_set])

        # matmul weights by data
        numerator = np.matmul(reference_set.T, weights)
        return numerator / weights.sum()

# Class that implements the dual tree mean-shift algorithm
# IDEA: Use a KDTree which holds references instead of copies.
    # - this way, we can change the actual values.
class DualTreeMeanShift():
    # __init__(): Constructor for MeanShift class, takes in the kernel for the algorithm
    # input: kernel to use in mean shift for kde
    def __init__(self, kernel, reference_set):
        if kernel == 'gaussian':
            self.kernel = utils.gaussian_kernel_ms
        if kernel == 'multivariate gaussian':
            self.kernel = utils.gaussian_kernel_correlations

        # save reference set
        self.reference_set = reference_set

        # create trees (query and reference are same at beginning)
        self.reference_tree = KDTree(self.reference_set.shape[1], self.reference_set)
        self.query_tree = KDTree(self.reference_set.shape[1], self.reference_set)

    # do(): Perform the dual tree mean shift algorithm to find the modes
    # input: bandwidth, error bound, some convergence limit
    # output: converged query set
    def do(self, bandwidth, error_bound, convergence_limit):
        # initialize distances array
        distances = [float('inf')] * len(self.reference_set)
        
        # go until max is greater than convergence limit
        while True: # needs to be changed to implement convergence (how ??)
            # find approximated m(x) vector
            m_x = self.ms_dualtree(query_tree.root, reference_tree.root)

            # rebuild query tree with new mean shift vector
            self.query_tree = KDTree(self.reference_set.shape[1], m_x)

        # query tree

    # ms_dualtree(): helper function for do, implements recursion for algorithm
    # input: Q node and R node
    # output: (when recursion is finished) approximated m(x) vector
    # side effect: rebuilds query tree
    def ms_dualtree(self, Q, R):
        pass

    # ms_dualtree_base(): base function for ms_dualtree()
    # input: Q node and R node
    # output: updates min and max bounds and updates x_q for the singular point
    def ms_dualtree_base(self, Q, R):
        pass

    # can_approximate(): function to ensure local pruning criterion
    # input: query node Q and reference node R
    # output: boolean whether we can approximate or not
    def can_approximate(self, Q, R):
        pass

    # approximate(): function to approximate an R node's contribution to numerator and denominator of m(x) 
    # input: query node Q and reference node R
    # output: update min and max bounds and updates x_q for all points in Q
    def approximate(self):
        pass

# Class that returns result from a mean-shift algorithm class MeanShiftResult:
    def __init__(self, reference_set, query_set):
        self.reference_set = reference_set
        self.query_set = query_set
