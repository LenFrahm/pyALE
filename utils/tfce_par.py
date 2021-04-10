from scipy import ndimage
from functools import reduce
import operator
import numpy as np

def tfce_par(invol, h, dh, voxel_dims=[2,2,2]):
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
    update_vals = []
    for E in np.arange(0.2,1,0.1):
        for H in np.arange(1.5,2.5,0.1):
            update_vals.append(np.multiply(np.power(h, H)*dh, np.power(szs, E)))
        
    return update_vals, mask