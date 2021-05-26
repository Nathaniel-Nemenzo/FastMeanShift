import time
import numpy as np
import utils

CONVERGENCE_LIMIT = 0.000001
GROUPING_DISTANCE = 0.00001

# Class that implements mean-shift algorithm
# TODO: Optimize
# optimization: http://proceedings.mlr.press/v2/wang07d/wang07d.pdf
class MeanShift:
    # __init__(): Constructor for MeanShift class, takes in the kernel for the algorithm
    def __init__(self, kernel, need_modes, need_groups):
        if kernel == 'gaussian':
            self.kernel = utils.gaussian_kernel_ms
        if kernel == 'multivariate gaussian':
            self.kernel = utils.gaussian_kernel_correlations
        self.need_modes = need_modes
        self.need_groups = need_groups

    # do(): Perform the mean shift algorithm to find the modes
    # data is of the form: [ features ]
    #                      [ features ] <- records
    def do(self, data, bandwidth):
        # Copy original points
        original_data = data

        # Convert data to np array
        working_data = np.array(data)

        # Keep track of which points have converged
        convergence_array = [False] * len(data)

        # when smallest diff of shift is less than convergence limit, all have converged
        counter = 0
        max_diff = float('inf')
        while max_diff > CONVERGENCE_LIMIT:
            max_diff = 0
            for point in range(len(working_data)):
                if not convergence_array[point]: 
                    temp_pt = working_data[point]

                    # compute mean shift for point
                    mean_shift = self.__compute_mean_shift(temp_pt, working_data, bandwidth) # compute m(x) - x
                    new_pt = temp_pt + mean_shift # x = m(x)

                    # check for convergence
                    diff = np.linalg.norm(new_pt - temp_pt)
                    if diff < CONVERGENCE_LIMIT:
                        convergence_array[point] = True

                    # get smallest diff
                    if diff > max_diff:
                        max_diff = diff

                    # reassign point
                    working_data[point] = new_pt

        # find modes 
        modes = self.__get_modes(working_data) if self.need_modes else None

        # put points into groups
        groups = self.__group_points(working_data, modes) if self.need_groups else None

        return MeanShiftResult(modes, original_data, working_data, groups)


    # __compute_mean_shift(): Computes mean shift vector m(x) - x 
    # per https://en.wikipedia.org/wiki/Mean_shift
    def __compute_mean_shift(self, x, data, bandwidth):
        # create weights array
        weights = np.array([self.kernel((np.linalg.norm(x - x_i) ** 2) / (bandwidth ** 2), bandwidth) for x_i in data])

        # matmul weights by data
        numerator = np.matmul(data.T, weights)
        return (numerator / weights.sum()) - x
        
    # __get_modes(): Find modes based on euclidian distance, all points should be converged before this is called
    def __get_modes(self, data):
        modes = []

        # append the first mode
        modes.append(data[0])
        for point in range(len(data)):
            add = True 
            for mode in modes:
                # there is already a group for that point
                if (np.linalg.norm(data[point] - mode) <= GROUPING_DISTANCE):
                    add = False
                    break
            if add:
                modes.append(data[point])
        return np.array(modes)

    # __group_points(): Groups points based on euclidian distance from modes, points must be converged before this is called
    def __group_points(self, data, modes):
        modes = np.array(modes)
        groups = [None] * len(data)
        for mode_index in range(len(modes)):
            for point in range(len(data)):
                if np.linalg.norm(data[point] - modes[mode_index]) <= GROUPING_DISTANCE:
                    groups[point] = mode_index
        return groups

# Class that returns result from a mean-shift algorithm
class MeanShiftResult:
    def __init__(self, modes, original_points, shifted_points, point_groups):
        self.modes = modes
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.groups = point_groups
