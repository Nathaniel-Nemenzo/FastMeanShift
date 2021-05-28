import time
import numpy as np
import matplotlib.pyplot as plt

from meanshift import MeanShift

def cluster(data):
    # create meanshift object
    ms = MeanShift('gaussian', data)

    # do meanshift
    start = time.time()
    results = ms.do(bandwidth = 2, convergence_limit = 0.00001)
    end = time.time()
    print("elapsed time: ", end - start)

    # get results
    query_set = results.query_set
    reference_set = results.reference_set

    # show plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # show scatter plot
    ax.scatter(x = reference_set[:, 0], y = reference_set[:, 1], s = 50, c = 'blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # print shifted points
    ax.scatter(x = query_set[:, 0], y = query_set[:, 1], s = 50, c = 'red', marker = 'X')
    fig.savefig("../fig")

data = np.genfromtxt('../test_data.csv', delimiter = ',')
cluster(data)
