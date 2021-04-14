import argparse
import pickle
import os
from os.path import isfile, isdir
import pandas as pd
import numpy as np
import nibabel as nb
import h5py
from utils.template import sample_space, affine
from utils.kernel import kernel_calc
from utils.compute import compute_ma, compute_hx, compute_hx_conv, compute_ale, compute_z, compute_tfce

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initates ALE Meta-Analysis.')
    parser.add_argument('file', type=str, help='Filename of excel file that stores study information (coordinates, etc.)')
    parser.add_argument('num_studies', type=int, help='Number of studies in the meta-analyses')
    parser.add_argument('num_true', type=int, help='Number of true activations in the meta-anylses')
    parser.add_argument('rep', type=int, help='Which sampling repitition to load data from.')
    parser.add_argument('-pt', nargs="?", default=False, help="Whether or not to run a TFCE parameter test. Can be True or False.")


    args = parser.parse_args()
    file = args.file
    num_studies = int(args.num_studies)
    num_true = int(args.num_true)
    rep = int(args.rep)
    parameter_test = args.pt
    if parameter_test == "True":
        parameter_test = True
    else:
        parameter_test = False
    
    
    s0 = list(range(num_studies))

    sim_data = h5py.File(f"{file}", "r")

    sample_sizes = sim_data[f"{num_studies}/{num_true}/{rep}/sample_sizes"][:]
    foci = []
    for key in sim_data[f"{num_studies}/{num_true}/{rep}/foci"].keys():
        foci.append(sim_data[f"{num_studies}/{num_true}/{rep}/foci"][key][:])
        
    num_foci = [foc.shape[0] for foc in foci]

    kernels = []
    for sample_size in sample_sizes:
        temp_uncertainty = 5.7/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))
        subj_uncertainty = (11.6/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))) / np.sqrt(sample_size)
        smoothing = np.sqrt(temp_uncertainty**2 + subj_uncertainty**2)
        kernels.append(kernel_calc(affine, smoothing, 31))

    mb = 1
    for i in s0:
        mb = mb*(1-np.max(kernels[i]))

    # define bins for histogram
    bin_steps=0.0001
    bin_edges = np.arange(0.00005,1-mb+0.001,bin_steps)
    bin_centers = np.arange(0,1-mb+0.001,bin_steps)
    step = int(1/bin_steps)

    ma = compute_ma(foci, kernels)
    hx = compute_hx(ma, bin_edges)
    ale = compute_ale(ma)
    hx_conv = compute_hx_conv(hx, bin_centers, step)
    z = compute_z(ale, hx_conv, step)
    tfce = compute_tfce(z, parameter_test=parameter_test)
    
    
    np.savez_compressed(f"Results/tmp/{num_studies}_{num_true}_{rep}", ale=ale, z=z, tfce=tfce, hx_conv=hx_conv, kernels=kernels, num_foci=num_foci)
    
    print(np.max(tfce))