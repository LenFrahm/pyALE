import numpy as np
import math
import pandas as pd
import numpy as np
# importing brain template shape
from utils.template import pad_shape

def kernel_calc(affine, fwhm, dims):
    s = (fwhm/2/math.sqrt(8*math.log(2)) + np.finfo(float).eps)**2 # fwhm -> sigma

    # 1D Gaussian based on sigma

    half_k_length = math.ceil(3.5*math.sqrt(s)) # Half of required kernel length
    x = list(range(-half_k_length, half_k_length + 1))
    oned_kernel = np.exp(-0.5 * np.multiply(x,x) / s) / math.sqrt(2*math.pi*s)
    oned_kernel = np.divide(oned_kernel, np.sum(oned_kernel))

    # Convolution of 1D Gaussians to create 3D Gaussian
    gkern3d = oned_kernel[:,None,None] * oned_kernel[None,:,None] * oned_kernel[None,None,:]

    
    # Padding to get matrix of desired size
    pad_size = int((dims - len(x)) / 2)
    gkern3d = np.pad(gkern3d, ((pad_size,pad_size),
                               (pad_size,pad_size),
                               (pad_size,pad_size)),'constant', constant_values=0)
    return gkern3d

def kernel_conv(peaks, kernel):
    data = np.zeros(pad_shape)
    for peak in peaks:
        x = (peak[0],peak[0]+31)
        y = (peak[1],peak[1]+31)
        z = (peak[2],peak[2]+31)
        data[x[0]:x[1], y[0]:y[1], z[0]:z[1]] = np.maximum(data[x[0]:x[1], y[0]:y[1], z[0]:z[1]], kernel)
    return data[15:data.shape[0]-15,15:data.shape[1]-15, 15:data.shape[2]-15]