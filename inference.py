import os
from os.path import isfile
import numpy as np
from scipy import ndimage
import argparse
import nibabel as nb
import pickle
from utils.compute import plot_and_save, compute_cluster
from utils.template import affine

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('clust_thresh', type=float)
    parser.add_argument('num_studies', type=int)
    parser.add_argument('num_true', type=int)
    parser.add_argument('rep', type=int)
    parser.add_argument('iter', type=int)
    parser.add_argument('-pt', nargs="?", default=False, help="Whether or not to run a TFCE parameter test. Can be True or False.")
    
    args = parser.parse_args()
    clust_thresh = args.clust_thresh
    num_studies = args.num_studies
    num_true = args.num_true
    rep = args.rep
    parameter_test = args.pt
    if parameter_test == "True":
        parameter_test = True
    else:
        parameter_test = False
    

    max_ale = []
    max_cluster = []
    max_tfce = []
    with open(f"logs/null/{num_studies}_{num_true}_{rep}.out", "r") as f:
        for line in f.readlines():
            split = line.split()
            if parameter_test:
                for idx, el in enumerate(split):
                    if idx == 0:
                        max_ale.append(float(el))
                    if idx == 1:
                        max_cluster.append(int(el))
                    if idx == 2 + param_iteration:
                        max_tfce.append(float(el.strip("[").strip("]").strip(",")))
            else:
                max_ale.append(float(split[0]))
                max_cluster.append(int(split[1]))
                max_tfce.append(float(split[2]))
                    
    npz = np.load(f"Results/tmp/{num_studies}_{num_true}_{rep}.npz")
    ale = npz["ale"]
    z = npz["z"]
    tfce = npz["tfce"]

    # cluster wise family wise error correction
    cut_cluster = np.percentile(max_cluster, 95)                  
    z, max_clust = compute_cluster(z, thresh=clust_thresh, cut_cluster=cut_cluster)

    # tfce error correction
    cut_tfce = np.percentile(max_tfce, 95)
    tfce = tfce*(tfce>cut_tfce)
    
    np.savez_compressed(f"Results/{num_studies}_{num_true}_{rep}", ale=ale, z=z, tfce=tfce)
    os.remove(f"Results/tmp/{num_studies}_{num_true}_{rep}.npz")