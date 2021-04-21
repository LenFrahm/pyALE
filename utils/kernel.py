import numpy as np
from math import sqrt, log, ceil, pi
import pandas as pd
import numpy as np
# importing brain template shape
from utils.template import pad_shape

def kernel_calc(affine, fwhm, dims):
    vx = np.sqrt(np.sum(affine[0:3,0:3]**2, axis=0))[0]
    s = (np.divide(np.divide(fwhm, vx), sqrt(8*log(2))) + np.finfo(float).eps)**2

    k_half = ceil(3.5*sqrt(s))

    x = list(range(-k_half, k_half + 1))
    k = np.exp(-0.5 * np.multiply(x,x) / s) / sqrt(2*pi*s)

    k = np.divide(k, sum(k))
    gkern3d = k[:,None,None] * k[None,:,None] * k[None,None,:]
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