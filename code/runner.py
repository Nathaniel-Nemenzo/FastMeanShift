import numpy as np
import time
from PIL import Image
from kdtree import KDTree

def test(data):
    # test data
    start = time.time()
    kdtree = KDTree(2, data) # two dimensional data (x, y)
    end = time.time()
    print("data size: ", len(data))
    print("test_data time elapsed: ", end - start)

    # smaller image
    im = Image.open('../photos/p0973lkk.jpeg', 'r')
    pixels = np.array(list(im.getdata()))
    start = time.time()
    kdtree = KDTree(3, pixels)
    end = time.time()
    print("data size: ", len(pixels))
    print("photo time elapsed: ", end - start)

    # big image
    im = Image.open('../photos/ntFmJUZ8tw3ULD3tkBaAtf.jpeg', 'r')
    pixels = np.array(list(im.getdata()))
    start = time.time()
    kdtree = KDTree(3, pixels)
    end = time.time()
    print("data size: ", len(pixels))
    print("photo time elapsed: ", end - start)

data = np.genfromtxt('../small_test_data.csv', delimiter = ',')
test(data)
