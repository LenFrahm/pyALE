import argparse
import pickle
import os
from os.path import isfile, isdir
import pandas as pd
import numpy as np
import nibabel as nb
from utils.template import sample_space
from utils.compute import illustrate_foci, compute_hx, compute_hx_conv, compute_ale, compute_z, compute_tfce, plot_and_save
from utils.compile_studies import compile_studies
from utils.folder_setup import folder_setup
from utils.read_exp_info import read_exp_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initates ALE Meta-Analysis.')
    parser.add_argument('path', type=str, help='working path')
    parser.add_argument('file', type=str, help='Filename of excel file that stores study information (coordinates, etc.)')
    parser.add_argument('type', type=str, help='Type of Analysis; check instructions file for possible types')
    parser.add_argument('name', type=str, help='Name given to analysis; used for provenance tracking')
    parser.add_argument('tags', type=str, nargs='+', help='Tags of experiments to include in analysis; check instructions file for syntax')

    args = parser.parse_args()
    
    path = args.path
    file = args.file
    type_ = args.type
    exp_name = args.name
    conditions = args.tags
    
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")
    
    if not os.path.isdir("Results"):
        os.mkdir("Results")

    if not os.path.isdir("Results/MainEffect/Full"):
        folder_setup(path, "MainEffect_Full")

    if isfile(f'Results/{file}.pickle'):
        with open(f'Results/{file}.pickle', 'rb') as f:
            exp_all, tasks = pickle.load(f)
    else:
        exp_all, tasks = read_exp_info(f'{file}')
        
    if conditions == []:
        exp_idxs = list(range(exp_all.shape[0]))
        exp_df = exp_all
    else:
        exp_idxs, masks, mask_names = compile_studies(conditions, tasks)
        exp_df = exp_all.loc[exp_idxs].reset_index(drop=True)

    s0 = list(range(exp_df.shape[0]))
    # highest possible ale value if every study had a peak at the same location.
    mb = 1
    for i in s0:
        mb = mb*(1-np.max(exp_df.at[i, 'Kernels']))
    
    # define bins for histogram
    bin_steps=0.0001
    bin_edges = np.arange(0.00005,1-mb+0.001,bin_steps)
    bin_centers = np.arange(0,1-mb+0.001,bin_steps)
    step = int(1/bin_steps)
    
    peaks = np.array([exp_df.XYZ[i].T for i in s0], dtype=object)    
    
    if not isfile(f'Results/MainEffect/Full/Volumes/Foci/{exp_name}.nii'):
        foci_arr = illustrate_foci(peaks)
        foci_arr = plot_and_save(foci_arr, img_folder=f'Results/MainEffect/Full/Images/Foci/{exp_name}.png', 
                                           nii_folder=f'Results/MainEffect/Full/Volumes/Foci/{exp_name}.nii')

    ma = np.array([exp_df.MA.values[i] for i in s0])
    hx = compute_hx(s0, ma, bin_edges)
    
    if isfile(f'Results/MainEffect/Full/NullDistributions/{exp_name}.pickle'):
        ale = nb.load(f'Results/MainEffect/Full/Volumes/ALE/{exp_name}.nii').get_fdata()
        with open(f'Results/MainEffect/Full/NullDistributions/{exp_name}.pickle', 'rb') as f:
            hx_conv, _ = pickle.load(f)    
    else:
        ale = compute_ale(ma)
        ale = plot_and_save(ale, img_folder=f'Results/MainEffect/Full/Images/ALE/{exp_name}.png',
                                 nii_folder=f'Results/MainEffect/Full/Volumes/ALE/{exp_name}.nii')
        hx_conv = compute_hx_conv(s0, hx, bin_centers, step)

        pickle_object = (hx_conv, hx)
        with open(f'Results/MainEffect/Full/NullDistributions/{exp_name}.pickle', "wb") as f:
            pickle.dump(pickle_object, f)
    
    if isfile(f'Results/MainEffect/Full/Volumes/TFCE/{exp_name}.nii'):
        z = nb.load(f'Results/MainEffect/Full/Volumes/Z/{exp_name}.nii').get_fdata()
        tfce = nb.load(f'Results/MainEffect/Full/Volumes/TFCE/{exp_name}.nii').get_fdata()
    else:
        z = compute_z(ale, hx_conv, step)
        tfce = compute_tfce(z, sample_space)

        z = plot_and_save(z, nii_folder=f'Results/MainEffect/Full/Volumes/Z/{exp_name}.nii')
        for i in range(18):
            out = plot_and_save(tfce[i], img_folder=f'Results/MainEffect/Full/Images/TFCE/{exp_name}_{i}.png',
                                       nii_folder=f'Results/MainEffect/Full/Volumes/TFCE/{exp_name}_{i}.nii')
    
    np.save(f"tmp/{exp_name}_num_peaks", exp_df.Peaks.values)
    np.save(f"tmp/{exp_name}_kernels", exp_df.Kernels.values)
    np.save(f"tmp/{exp_name}_hx_conv", hx_conv)
    
    pickle_object = (exp_df.Subjects, exp_df.MA, exp_df.Author, exp_idxs, tasks)
    with open(f'tmp/{exp_name}_contribution.pickle', "wb") as f:
        pickle.dump(pickle_object, f)