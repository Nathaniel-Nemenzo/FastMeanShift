import time
import numpy as np
import matplotlib.pyplot as plt

from meanshift import MeanShift

def cluster(data):
    # create meanshift object
    ms = MeanShift('gaussian', True, True)

    # do meanshift
    start = time.time()
    results = ms.do(data, bandwidth = 2)
    end = time.time()
    print("elapsed time: ", end - start)

    # get results
    original_points = results.original_points
    shifted_points = results.shifted_points
    modes = results.modes
    groups = results.groups

    # show plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # show scatter plot
    ax.scatter(x = original_points[:, 0], y = original_points[:, 1], s = 50, c = groups)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # print shifted points
    ax.scatter(x = shifted_points[:, 0], y = shifted_points[:, 1], s = 50, c = 'red', marker = 'X')
    fig.savefig("ms result_working")


data = np.genfromtxt('data.csv', delimiter = ',')
cluster(data)
