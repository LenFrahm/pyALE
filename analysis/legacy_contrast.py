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
from utils.compute import compute_ale_diff, compute_perm_diff, compute_sig_diff, plot_and_save
# importing brain template information
from utils.template import shape, pad_shape, prior, affine

EPS = np.finfo(float).eps # float precision

def legacy_contrast(exp_dfs, exp_names, diff_thresh=0.05, null_repeats=10000, masking=True):
    
    ma = [np.stack(exp_dfs[i].MA.values) for i in (0,1)]
    
    fx1 = nb.load(f"Results/MainEffect/Full/Volumes/Corrected/{exp_names[0]}_cFWE05.nii").get_fdata()
    fx2 = nb.load(f"Results/MainEffect/Full/Volumes/Corrected/{exp_names[1]}_cFWE05.nii").get_fdata()

    thresh_in_percent = int((1-diff_thresh)*100)
    # Check if contrast has already been calculated
    if isfile(f"Results/Contrast/Full/{exp_names[0]}--{exp_names[1]}_P{thresh_in_percent}.nii"):
        print(f"{exp_names[0]} x {exp_names[1]} - Loading contrast.")
        contrast_arr = nb.load(f"Results/Contrast/Full/{exp_names[0]}--{exp_names[1]}_P{thresh_in_percent}.nii").get_fdata()
    else:
        print(f"{exp_names[0]} x {exp_names[1]} - Computing positive contrast.")
        s = [list(range(exp_dfs[i].shape[0])) for i in (0,1)]
        if masking == True:
            mask = fx1 > 0
        else:
            mask = prior
        if mask.sum() > 0:
            ale_diff = compute_ale_diff(s, ma, mask)
            masked_ma = 1 - np.vstack((ma[0][:,mask], ma[1][:,mask]))
            # estimate null distribution of difference values if studies would be randomly assigned to either meta analysis
            perm_diff = Parallel(n_jobs=-1)(delayed(compute_perm_diff)(s, masked_ma) for i in range(null_repeats))
            z1, sig_idxs1 = compute_sig_diff(fx1, mask, ale_diff, perm_diff, null_repeats, diff_thresh)

        else:
            print(f"{exp_names[0]}: No significant indices!")
            z1, sig_idxs1 = [], []


        print(f"{exp_names[0]} x {exp_names[1]} - Computing negative contrast.")
        s = [list(range(exp_dfs[i].shape[0])) for i in (1,0)]
        if masking == True:
            mask = fx2 > 0
        else:
            mask = prior
        if mask.sum() > 0:
            ale_diff = compute_ale_diff(s, ma, mask)
            masked_ma = 1 - np.vstack((ma[1][:,mask], ma[0][:,mask]))
            # estimate null distribution of difference values if studies would be randomly assigned to either meta analysis
            perm_diff = Parallel(n_jobs=-1)(delayed(compute_perm_diff)(s, masked_ma) for i in range(null_repeats))
            z2, sig_idxs2 = compute_sig_diff(fx2, mask, ale_diff, perm_diff, null_repeats, diff_thresh)
            

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
        conj_arr = np.minimum(fx1, fx2)
        if conj_arr is not None:
            conj_arr = plot_and_save(conj_arr, img_folder=f"Results/Contrast/Full/Conjunctions/Images/{exp_names[0]}_AND_{exp_names[1]}_cFWE.png",
                                               nii_folder=f"Results/Contrast/Full/Conjunctions/{exp_names[0]}_AND_{exp_names[1]}_cFWE.nii")
            
    print(f"{exp_names[0]} & {exp_names[1]} - done!")