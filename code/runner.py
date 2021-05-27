import numpy as np
from kdtree import KDTree

def test(data):
    kdtree = KDTree(2, data) # two dimensional data (x, y)

data = np.genfromtxt('../test_data.csv', delimiter = ',')
test(data)
