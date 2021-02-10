from scipy import ndimage
from functools import reduce
import operator
import numpy as np

def tfce_par(invol, h, voxel_dims=[2,2,2], dh=0.1, E=0.6, H=2):
    thresh = np.array(invol > h)
    #look for suprathreshold clusters
    labels, cluster_count = ndimage.label(thresh)
    if cluster_count > 0: 
        #calculate the size of the cluster; first voxel count, then multiplied with the voxel volume in mm
        _ , sizes = np.unique(labels, return_counts=True)
        sizes[0] = 0 
        sizes = sizes * reduce(operator.mul, voxel_dims)
        #mask out labeled areas to not perform tfce calculation on the whole brain
        mask = labels > 0
        szs = sizes[labels[mask]]
        update_vals = np.multiply(np.power(h, H), np.power(szs, E))
    return update_vals, mask