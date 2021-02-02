import numpy as np
from scipy import ndimage
from tfce import tfce_par
from scipy.stats import norm
from joblib import Parallel, delayed
from mytime import tic, toc

def simulate_noise(sample_space,
                   s0,
                   num_peaks,
                   kernels,
                   c_null,
                   eps=np.finfo(float).eps,
                   uc = 0.001,
                   tfce_params = [0.1, 0.6, 2],
                   template_shape = [91,109,91],
                   pad_tmp_shape=[121, 139, 121],
                   voxel_dims=[2,2,2],
                   step=10000):
    
    sample_space_size = sample_space.shape[1]
    vx = np.zeros((len(s0), sample_space_size))

    for i, s in enumerate(s0):
        sample_peaks = sample_space[:,np.random.randint(0,sample_space_size, num_peaks[s])]
        data = np.zeros((pad_tmp_shape))
        for peak in sample_peaks.T:
            x_range = (peak[0],peak[0]+31)
            y_range = (peak[1],peak[1]+31)
            z_range = (peak[2],peak[2]+31)
            data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] = \
                np.maximum(data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]],
                           kernels[s])
        vx[i,:] = data[sample_space[0]+15, sample_space[1]+15, sample_space[2]+15]

    data = 1-np.prod(1-vx, axis=0)

    # Peak ALE threshold
    nm = np.max(data)
    # Cluster level threshold
    ale_step = np.round(data*step).astype(int)
    p = np.array([c_null[i] for i in ale_step])
    p[p < eps] = eps

    sig_coords = sample_space[:,p < uc]
    sig_arr = np.zeros(template_shape)
    sig_arr[tuple(sig_coords)] = 1
    labels, cluster_count = ndimage.label(sig_arr)
    if cluster_count == 1:
        nn = 0
    else:
        nn = np.max(np.bincount(labels[labels>0]))
        
    z = np.zeros(template_shape)
    z[sample_space[0], sample_space[1], sample_space[2]] = norm.ppf(1-p)
    
    # TFCE threshold
    tfce_arr = np.zeros(z.shape)
    dh, H, E = tfce_params
    vals, masks = zip(*Parallel(n_jobs=-1, backend="threading")(delayed(tfce_par)(invol=z, h=h, dh=dh) for h in np.arange(0, np.max(z), dh)))
    for i in range(len(vals)):
        tfce_arr[masks[i]] += vals[i]
    nt = np.max(tfce_arr)
    return nm, nn, nt