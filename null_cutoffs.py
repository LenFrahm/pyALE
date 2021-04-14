import numpy as np
from utils.compute import compute_null_cutoffs
from utils.template import sample_space, shape
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Calculates null cutoffs from randomly intiated coordinates')
    parser.add_argument('clust_thresh', type=float)
    parser.add_argument('num_studies', type=int)
    parser.add_argument('num_true', type=int)
    parser.add_argument('rep', type=int)
    parser.add_argument('iter', type=int)
    parser.add_argument('-pt', nargs="?", default=False, help="Whether or not to run a TFCE parameter test. Can be True or False.")
    
    args = parser.parse_args()
    clust_thresh = float(args.clust_thresh)
    num_studies = args.num_studies
    num_true = args.num_true
    rep = args.rep
    iter_ = int(args.iter)
    parameter_test = args.pt
    if parameter_test == "True":
        parameter_test = True
    else:
        parameter_test = False
    
    npz = np.load(f"Results/tmp/{num_studies}_{num_true}_{rep}.npz")
    
    num_foci = npz["num_foci"]
    kernels = npz["kernels"]
    hx_conv = npz["hx_conv"]

    for i in range(iter_):   
        null_ale, null_max_ale, null_max_cluster, null_max_tfce = compute_null_cutoffs(sample_space,
                                                                                       num_foci,
                                                                                       kernels,
                                                                                       hx_conv=hx_conv,
                                                                                       thresh=clust_thresh,
                                                                                       tfce=1,
                                                                                       parameter_test=parameter_test)
        print(null_max_ale, null_max_cluster, null_max_tfce)