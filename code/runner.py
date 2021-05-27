import numpy as np
import time

from PIL import Image
from kdtree import KDTree

def test(data):
    # test data
    start = time.time()
    kdtree = KDTree(2, data) # two dimensional data (x, y)
    print("number of nodes: ", kdtree.num_nodes)
    print("tree: ")
    kdtree.print_tree()

    end = time.time()
    print("test_data time elapsed: ", end - start)
    quit()

    # image
    start = time.time()
    im = Image.open('../photos/p0973lkk.jpeg', 'r')
    pixels = list(im.getdata())
    kdtree = KDTree(3, np.array(pixels))
    end = time.time()
    print("photo time elapsed: ", end - start)

data = np.genfromtxt('../test_data.csv', delimiter = ',')
test(data)
