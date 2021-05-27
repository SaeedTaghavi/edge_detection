# -- coding: utf-8 --
"""
Created on Wed May 26 13:11:14 2021

@author: Saeed Taghavi
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Read Images
img = mpimg.imread('./img/sample0.jpg')

# Output Images
plt.imshow(img)
#get image dimensions

print(np.shape(img))

print(img.ndim)
if (img.ndim==2):
    img=img
elif(img.ndim==3):
    height, width , depth = np.shape(img)

    # seprate different layers
    R,G,B = img[:,:,0],img[:,:,1],img[:,:,2]
    img =R

from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter

#smoth image with gaussian_filter
result = gaussian_filter(img, sigma=2)
# result =img
#define different kernels
kernelLaplacian =[[0,-1,0],[-1,4,-1],[0,-1,0]]
kernelSobely =[[1,2,1],[0,0,0],[-1,-2,-1]]
kernelSobelx = [[1,0,-1],[2,0,-2],[1,0,-1]]

#edge detection using a kernel
new = convolve(result, kernelLaplacian)
plt.imshow(new)
