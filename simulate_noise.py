import numpy as np
from scipy import ndimage
from tfce import tfce
from scipy.stats import norm
import timeit

def simulate_noise(sample_space,
                   s0,
                   num_peaks,
                   kernels,
                   c_null,
                   eps=np.finfo(float).eps,
                   uc = 0.001,
                   tfce_params = [0.1, 0.6, 2],
                   pad_tmp_shape=[121, 139, 121],
                   voxel_dims=[2,2,2],
                   step=10000):
    starttime = timeit.default_timer()
    sample_space_size = sample_space.shape[1]
    vx = np.zeros((len(s0), sample_space_size))
    data = np.zeros((pad_tmp_shape))

    for i, s in enumerate(s0):
        sample_peaks = sample_space[:,np.random.randint(0,sample_space_size, num_peaks[s])]
        for peak in sample_peaks.T:
            x_range = (peak[0],peak[0]+31)
            y_range = (peak[1],peak[1]+31)
            z_range = (peak[2],peak[2]+31)
            data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] = \
                np.maximum(data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]],
                           kernels[s])

        vx[i,:] = data[sample_space[0], sample_space[1], sample_space[2]]
    
    data = 1-np.prod(1-vx, axis=0)

    # Peak ALE threshold
    nm = np.max(data)
    print("time taken for ale calc:" + str(timeit.default_timer() - starttime))
    starttime = timeit.default_timer()
    # Cluster level threshold
    palette, index = np.unique(np.round(data*step), return_inverse=True)
    index = palette[index].astype(int)
    p = c_null[index].reshape(data.shape)
    p[p < eps] = eps
    labels, cluster_count = ndimage.label(np.array(p < uc))
    nn = np.max(np.bincount(labels[labels>0]))

    z = np.zeros(pad_tmp_shape)
    z[sample_space[0], sample_space[1], sample_space[2]] = norm.ppf(1-p)
    z = z[15:z.shape[0]-15,15:z.shape[1]-15, 15:z.shape[2]-15]
    print("time taken for p-value calc:" + str(timeit.default_timer() - starttime))
    starttime = timeit.default_timer()
    # TFCE threshold
    tfce_arr = tfce(invol=z,
                    voxel_dims=voxel_dims,
                    dh=tfce_params[0],
                    E=tfce_params[1],
                    H=tfce_params[2])
    nt = np.max(tfce_arr)
    print("time taken for tfce calc:" + str(timeit.default_timer() - starttime))
    
    return nm, nn, nt