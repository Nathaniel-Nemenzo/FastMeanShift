import numpy as np
import time

from PIL import Image
from kdtree import KDTree

def test(data):
    # test data
    start = time.time()
    kdtree = KDTree(2, data) # two dimensional data (x, y)
    end = time.time()
    print("test_data time elapsed: ", end - start)

    # BIG image
    start = time.time()
    im = Image.open('../photos/p0973lkk.jpeg', 'r')
    pixels = list(im.getdata())
    print(len(pixels))
    kdtree = KDTree(3, np.array(pixels))
    end = time.time()
    print("test_data time elapsed: ", end - start)


data = np.genfromtxt('../test_data.csv', delimiter = ',')
test(data)
