from scipy import ndimage
from functools import reduce
import operator
import numpy as np
from math import pow

def tfce(invol, voxel_dims, dh=0.1, E=0.5, H=2.0, negative=False):
    #create empty output volume
    outvol = np.zeros(invol.shape)
    #loop over range of heights between 0 and the maximum, with a stepsize specified by dh
    for h in np.arange(0, np.max(invol), dh):
        thresh = np.array(invol > h)
        #look for suprathreshold clusters
        labels, cluster_count = ndimage.label(thresh)
        if cluster_count > 0: 
            
            #calculate the size of the cluster; first voxel count, then multiplied with the voxel volume in mm
            sizes = np.array(ndimage.sum(thresh, labels, list(range(cluster_count+1))))
            sizes = sizes * reduce(operator.mul, voxel_dims)
            
            #mask out labeled areas to not perform tfce calculation on the whole brain
            mask = labels > 0
            szs = sizes[labels[mask]]
            update_vals = (pow(h, H) * dh) * np.power(szs, E)

            if negative:
                outvol[mask] -= update_vals
            else:
                outvol[mask] += update_vals

    return outvol