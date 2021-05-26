import time
import numpy as np
import matplotlib.pyplot as plt

from meanshift import MeanShift
from PIL import Image

def segment(pixels):
    # create mean shift object
    ms = MeanShift('gaussian', False, False)

    # get shifted points 
    results = ms.do(pixels, 25)
    new_pixels = results.shifted_points
    print(new_pixels)

    return Image.fromarray(new_pixels)

img = Image.open('photos/mean_shift_image.jpeg', 'r')
pixels = np.array(list(img.getdata()))
print(pixels.shape)
new_img = segment(pixels)
new_img.save('new.png')

