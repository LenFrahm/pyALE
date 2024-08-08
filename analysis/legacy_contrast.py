import os
from os.path import isfile
import nibabel as nb
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from nilearn import plotting
from scipy import ndimage
from utils.kernel import kernel_conv
from utils.compute import compute_ale, compute_perm_diff, compute_sig_diff, plot_and_save
# importing brain template information
from utils.template import shape, pad_shape, prior, affine

EPS = np.finfo(float).eps # float precision

def legacy_contrast(exp_dfs, exp_names, diff_thresh=0.05, null_repeats=10000, nprocesses=4):

    group1_num_exp = exp_dfs[0].shape[0]
    ma_group1 = np.stack(exp_dfs[0].MA.values)
    ale_group1 = compute_ale(ma_group1)

    group2_num_exp = exp_dfs[1].shape[0]
    ma_group2 = np.stack(exp_dfs[1].MA.values)
    ale_group2 = compute_ale(ma_group2)
    
    ale_difference = ale_group1 - ale_group2

    thresh_in_percent = int((1-diff_thresh)*100)
    # Check if contrast has already been calculated
    if isfile(f"Results/Contrast/Full/{exp_names[0]}--{exp_names[1]}_P{thresh_in_percent}.nii"):
        print(f"{exp_names[0]} x {exp_names[1]} - Loading contrast.")
        contrast_arr = nb.load(f"Results/Contrast/Full/{exp_names[0]}--{exp_names[1]}_P{thresh_in_percent}.nii").get_fdata()
    else:
        print(f"{exp_names[0]} x {exp_names[1]} - Computing positive contrast.")
        group1_main_effect = nb.load(f"Results/MainEffect/Full/Volumes/Corrected/{exp_names[0]}_cFWE05.nii").get_fdata()
        mask = group1_main_effect > 0
        if mask.sum() > 0:
            stacked_masked_ma = np.vstack((ma_group1[:,mask], ma_group2[:,mask]))
            # estimate null distribution of difference values if studies would be randomly assigned to either meta analysis
            perm_diffs = Parallel(n_jobs=nprocesses)(delayed(compute_perm_diff)(group1_num_exp, group2_num_exp, stacked_masked_ma) for i in range(null_repeats))
            z1, sig_idxs1 = compute_sig_diff(ale_difference[mask], perm_diffs, mask, diff_thresh, direction='positive')

        else:
            print(f"{exp_names[0]}: No significant indices!")
            z1, sig_idxs1 = [], []


        print(f"{exp_names[0]} x {exp_names[1]} - Computing negative contrast.")
        group2_main_effect = nb.load(f"Results/MainEffect/Full/Volumes/Corrected/{exp_names[1]}_cFWE05.nii").get_fdata()
        mask = group2_main_effect > 0
        if mask.sum() > 0:
            stacked_masked_ma = np.vstack((ma_group1[:,mask], ma_group2[:,mask]))
            # estimate null distribution of difference values if studies would be randomly assigned to either meta analysis
            perm_diffs = Parallel(n_jobs=nprocesses)(delayed(compute_perm_diff)(group1_num_exp, group2_num_exp, stacked_masked_ma) for i in range(null_repeats))
            z2, sig_idxs2 = compute_sig_diff(ale_difference[mask], perm_diffs, mask, diff_thresh, direction='negative')
            

        else:
            print(f"{exp_names[1]}: No significant indices!")
            z2, sig_idxs2 = [], []

        print(f"{exp_names[0]} x {exp_names[1]} - Inference and printing.")
        contrast_arr = np.zeros(shape)
        contrast_arr[tuple(sig_idxs1)] = z1
        contrast_arr[tuple(sig_idxs2)] = -z2
        contrast_arr = plot_and_save(contrast_arr, img_folder=f"Results/Contrast/Full/Images/{exp_names[0]}--{exp_names[1]}_P{thresh_in_percent}.png",
                                                   nii_folder=f"Results/Contrast/Full/{exp_names[0]}--{exp_names[1]}_P{thresh_in_percent}.nii")
    
    #Check if conjunction has already been calculated
    if isfile(f"Results/Contrast/Full/Conjunctions/{exp_names[0]}_AND_{exp_names[1]}_cFWE.nii"):
        print(f"{exp_names[0]} & {exp_names[1]} - Loading conjunction.")
        conj_arr = nb.load(f"Results/Contrast/Full/Conjunctions/{exp_names[0]}_AND_{exp_names[1]}_cFWE.nii").get_fdata()
    else:
        print(f"{exp_names[0]} & {exp_names[1]} - Computing conjunction.")
        conj_arr = np.minimum(group1_main_effect, group2_main_effect)
        if conj_arr is not None:
            conj_arr = plot_and_save(conj_arr, img_folder=f"Results/Contrast/Full/Conjunctions/Images/{exp_names[0]}_AND_{exp_names[1]}_cFWE.png",
                                               nii_folder=f"Results/Contrast/Full/Conjunctions/{exp_names[0]}_AND_{exp_names[1]}_cFWE.nii")
            
    print(f"{exp_names[0]} & {exp_names[1]} - done!")