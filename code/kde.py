import numpy as np
import utils

class KernelDensityEstimation:
    def __init__(self, kernel):
        if kernel == 'gaussian':
            self.kernel = utils.gaussian_kernel_kde

    def do(self, x, data, bandwidth):
        denominator = data.shape[0] * (bandwidth ** x.shape[0])
        numerator = 0
        for x_i in data:
            numerator += self.kernel((x - x_i) / bandwidth)
        return numerator / denominator
