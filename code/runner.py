import numpy as np
import time

from PIL import Image
from kdtree import KDTree

def test(data):
    # test data
    start = time.time()

    kdtree = KDTree(2, data) # two dimensional data (x, y)
    print("number of nodes: ", kdtree.num_nodes)
    print("number of datum: ", kdtree.num_datum)
    end = time.time()

    print("test_data time elapsed: ", end - start)

    # image
    start = time.time()
    im = Image.open('../photos/p0973lkk.jpeg', 'r')

    kdtree = KDTree(3, np.array(list(im.getdata())))
    print("number of nodes: ", kdtree.num_nodes)
    print("number of datum: ", kdtree.num_datum)

    end = time.time()
    print("photo time elapsed: ", end - start)

data = np.genfromtxt('../test_data.csv', delimiter = ',')
test(data)
