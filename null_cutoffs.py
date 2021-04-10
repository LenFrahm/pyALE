import numpy as np
from utils.compute import compute_null_cutoffs
from utils.template import sample_space, shape
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculates null cutoffs from randomly intiated coordinates')
    parser.add_argument('exp_name', type=str)
    parser.add_argument('clust_thresh', type=float)
    args = parser.parse_args()
    exp_name = args.exp_name
    clust_thresh = args.clust_thresh
    

    num_peaks = np.load(f"tmp/{exp_name}_num_peaks.npy")
    kernels = np.load(f"tmp/{exp_name}_kernels.npy", allow_pickle=True)
    hx_conv = np.load(f"tmp/{exp_name}_hx_conv.npy")
    
    s0 = list(range(kernels.shape[0]))

    for i in range(100):   
        null_ale, null_max_ale, null_max_cluster, null_max_tfce = compute_null_cutoffs(s0,
                                                                                       sample_space,
                                                                                       num_peaks,
                                                                                       kernels,
                                                                                       hx_conv=hx_conv,
                                                                                       thresh=clust_thresh,
                                                                                       tfce=1)
        print(null_max_ale, null_max_cluster, null_max_tfce)