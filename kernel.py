import numpy as np
from recordclass import recordclass
from math import sqrt, log, ceil, pi
import pandas as pd
import numpy as np

def kernel_calc(affine, fwhm, dims):
    #Conversion from fwhm to sigma based on affine
    vx = np.sqrt(np.sum(affine[0:3,0:3]**2, axis=0))
    s = (np.divide(np.divide(fwhm, vx), sqrt(8*log(2))) + np.finfo(float).eps)**2
    
    # 1D Gaussian based on sigma
    Dim = recordclass('Dim', ['s', 'k'])
    x = Dim(s=0,k=0)
    y = Dim(s=0,k=0)
    z = Dim(s=0,k=0)
    r = (x,y,z)

    for i in range(3):
        r[i].s = ceil(3.5*sqrt(s[i])) # Half of required kernel length
        x = list(range(-r[i].s, r[i].s + 1))
        r[i].k = np.exp(-0.5 * np.multiply(x,x) / s[i]) / sqrt(2*pi*s[i])
        r[i].k = np.divide(r[i].k, sum(r[i].k))
    
    # Convolution of 1D Gaussians to create 3D Gaussian
    gkern3d = r[0].k[:,None,None] * r[1].k[None,:,None] * r[2].k[None,None,:]
    
    # Padding to get matrix of desired size
    pad_size = int((dims - len(x)) / 2)
    gkern3d = np.pad(gkern3d, ((pad_size,pad_size),
                               (pad_size,pad_size),
                               (pad_size,pad_size)),'constant', constant_values=0)
    return gkern3d

def kernel_conv(i, df, shape):
    data = np.zeros(shape)
    for ii in range(df.at[i, 'Peaks']):
                coords = df.XYZ[i].T[:,:3][ii]
                x_range = (coords[0],coords[0]+31)
                y_range = (coords[1],coords[1]+31)
                z_range = (coords[2],coords[2]+31)
                data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] = \
                np.maximum(data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]],
                           df.at[i, 'Kernel'])
    return data[15:data.shape[0]-15,15:data.shape[1]-15, 15:data.shape[2]-15]