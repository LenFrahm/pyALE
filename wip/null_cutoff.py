from scipy import ndimage
from scipy.stats import norm
from functools import reduce
import operator
import numpy as np
from template import shape, pad_shape
from joblib import Parallel, delayed

EPS =  np.finfo(float).eps

def kernel_conv(peaks, kernel):
    data = np.zeros(pad_shape)
    for peak in peaks:
        x = (peak[0],peak[0]+31)
        y = (peak[1],peak[1]+31)
        z = (peak[2],peak[2]+31)
        data[x[0]:x[1], y[0]:y[1], z[0]:z[1]] = np.maximum(data[x[0]:x[1], y[0]:y[1], z[0]:z[1]], kernel)
    return data[15:data.shape[0]-15,15:data.shape[1]-15, 15:data.shape[2]-15]

def tfce_par(invol, h, dh, voxel_dims=[2,2,2], E=0.6, H=2):
    thresh = np.array(invol > h)
    #look for suprathreshold clusters
    labels, cluster_count = ndimage.label(thresh)
    #calculate the size of the cluster; first voxel count, then multiplied with the voxel volume in mm
    _ , sizes = np.unique(labels, return_counts=True)
    sizes[0] = 0 
    sizes = sizes * reduce(operator.mul, voxel_dims)
    #mask out labeled areas to not perform tfce calculation on the whole brain
    mask = labels > 0
    szs = sizes[labels[mask]]
    update_vals = np.multiply(np.power(h, H)*dh, np.power(szs, E))
        
    return update_vals, mask

def compute_null_cutoffs(s0, sample_space, num_peaks, kernels, hx_conv, step=10000, thresh=0.001):
    
    null_peaks = np.array([sample_space[:,np.random.randint(0,
                                                            sample_space.shape[1],
                                                            num_peaks[i])].T for i in s0], dtype=object)
    
    ma = np.zeros((len(s0), shape[0], shape[1], shape[2]))
    for i, s in enumerate(s0):
        ma[i, :] = kernel_conv(peaks = null_peaks[i], 
                               kernel = kernels[s])
        
    ale = 1-np.prod(1-ma, axis=0)
    max_ale = np.max(ale)
    
    ale_step = np.round(ale*step).astype(int)
    p = np.array([hx_conv[i] for i in ale_step])
    p[p < EPS] = EPS
    z = norm.ppf(1-p)
    
    sig_arr = np.zeros(shape)
    sig_arr[z > norm.ppf(1-thresh)] = 1
    labels, cluster_count = ndimage.label(sig_arr)
    max_clust = np.max(np.bincount(labels[labels>0]))
        
    lc = np.min(sample_space, axis=1)
    uc = np.max(sample_space, axis=1)  
    z = z[lc[0]:uc[0],lc[1]:uc[1],lc[2]:uc[2]]
    
    delta_t = np.max(z)/100
    
    tfce = np.zeros(z.shape)
    # calculate tfce values using the parallelized function
    vals, masks = zip(*Parallel(n_jobs=-1)
                     (delayed(tfce_par)(invol=z, h=h, dh=delta_t) for h in np.arange(0, np.max(z), delta_t)))
    # Parallelization makes it necessary to integrate the results afterwards
    # Each repition creats it's own mask and an amount of values corresponding to that mask
    for i in range(len(vals)):
        tfce[masks[i]] += vals[i]
    
    max_tfce = np.max(tfce)
    
    return ale, max_ale, max_clust, max_tfce

def par_null(s0, sample_space, num_peaks, kernels, hx_conv, null_repeats):
    
    ale, max_ale, max_cluster, max_tfce = zip(*Parallel(n_jobs=-1, verbose=1)(delayed(
                                          compute_null_cutoffs)(s0=s0,
                                                                sample_space=sample_space,
                                                                num_peaks=num_peaks,
                                                                kernels=kernels,
                                                                hx_conv=hx_conv) for i in range(null_repeats)))
    return ale, max_ale, max_cluster, max_tfce