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

    # smaller image
    im = Image.open('../photos/p0973lkk.jpeg', 'r')
    start = time.time()
    kdtree = KDTree(3, np.array(list(im.getdata())))
    end = time.time()
    print("photo time elapsed: ", end - start)

    # big image
#    im = Image.open('../photos/ntFmJUZ8tw3ULD3tkBaAtf.jpeg', 'r')
#    start = time.time()
#    kdtree = KDTree(3, np.array(list(im.getdata())))
#    end = time.time()
#    print("photo time elapsed: ", end - start)

data = np.genfromtxt('../small_test_data.csv', delimiter = ',')
test(data)
