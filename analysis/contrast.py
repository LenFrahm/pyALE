import os
from os.path import isfile
import nibabel as nb
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
import pickle
from utils.template import prior, affine, shape, pad_shape, sample_space
from utils.compute import plot_and_save, compute_ale_diff, compute_null_diff


def contrast(exp_dfs, exp_names, null_repeats=1000, target_n="Full", diff_repeats=1000, nprocesses=2):
    
    s = [list(range(exp_dfs[i].shape[0])) for i in (0,1)]   
    ma = [np.stack(exp_dfs[i].MA.values) for i in (0,1)]
    
    if target_n == "Full":
        cat = "Full"
        fx1 = nb.load(f"Results/MainEffect/Full/Volumes/Corrected/{exp_names[0]}_cFWE05.nii").get_fdata()
        fx2 = nb.load(f"Results/MainEffect/Full/Volumes/Corrected/{exp_names[1]}_cFWE05.nii").get_fdata()
    else:
        cat = "Balanced"
        fx1 = nb.load(f"Results/MainEffect/CV/Volumes/{exp_names[0]}_{target_n}.nii").get_fdata()
        fx2 = nb.load(f"Results/MainEffect/CV/Volumes/{exp_names[1]}_{target_n}.nii").get_fdata()
        
    if not isfile(f"Results/Contrast/{cat}/Conjunctions/{exp_names[0]}_AND_{exp_names[1]}_{target_n}.nii"):
        print(f'{exp_names[0]} x {exp_names[1]} - computing conjunction')
        conjunction = np.minimum(fx1, fx2)
        conjunction = plot_and_save(conjunction, img_folder=f"Results/Contrast/{cat}/Conjunctions/Images/{exp_names[0]}_AND_{exp_names[1]}_{target_n}.png",
                                                 nii_folder=f"Results/Contrast/{cat}/Conjunctions/{exp_names[0]}_AND_{exp_names[1]}_{target_n}.nii")
            
    if isfile(f"Results/Contrast/{cat}/NullDistributions/{exp_names[0]}_x_{exp_names[1]}_{target_n}.pickle"):
        print(f'{exp_names[0]} x {exp_names[1]} - loading actual diff and null extremes')
        with open(f"Results/Contrast/{cat}/NullDistributions/{exp_names[0]}_x_{exp_names[1]}_{target_n}.pickle", 'rb') as f:
            r_diff, prior, min_diff, max_diff = pickle.load(f)
    else:
        print(f'{exp_names[0]} x {exp_names[1]} - computing actual diff and null extremes')
        prior = np.zeros((91,109,91)).astype(bool)
        prior[tuple(sample_space)] = 1
        
        if cat == "Balanced":
            r_diff = Parallel(n_jobs=nprocesses, verbose=1)(delayed(compute_ale_diff)(s, ma, prior, target_n) for i in range(diff_repeats))
            r_diff = np.mean(r_diff, axis=0)   
            min_diff, max_diff = zip(*Parallel(n_jobs=nprocesses, verbose=1)(delayed(compute_null_diff)(s, prior, exp_dfs, target_n, diff_repeats) for i in range(null_repeats)))

        else:
            r_diff = compute_diff(s, ma, prior)
            min_diff, max_diff = zip(*Parallel(n_jobs=nprocesses, verbose=1)(delayed(compute_null_diff)(s, prior, exp_dfs) for i in range(null_repeats)))
        
        pickle_object = (r_diff, prior, min_diff, max_diff)
        with open(f"Results/Contrast/{cat}/NullDistributions/{exp_names[0]}_x_{exp_names[1]}_{target_n}.pickle", "wb") as f:
            pickle.dump(pickle_object, f)

    if not isfile(f"Results/Contrast/{cat}/{exp_names[0]}_x_{exp_names[1]}_{target_n}_FWE05.nii"):
        print(f'{exp_names[0]} x {exp_names[1]} - computing significant contrast')
        min_cut, max_cut = np.percentile(min_diff, 2.5), np.percentile(max_diff, 97.5)  

        sig_diff = r_diff * np.logical_or(r_diff < min_cut, r_diff > max_cut)
        sig_diff[sig_diff > 0] = np.array([-1 * norm.ppf((sum(max_diff >= diff)+1) / (null_repeats+1)) for diff in sig_diff[sig_diff > 0]])
        sig_diff[sig_diff < 0] = np.array([norm.ppf((sum(min_diff <= diff)+1) / (null_repeats+1)) for diff in sig_diff[sig_diff < 0]])

        brain_sig_diff = np.zeros(shape)
        brain_sig_diff[prior] = sig_diff

        sig_diff2 = plot_and_save(brain_sig_diff, img_folder=f"Results/Contrast/{cat}/Images/{exp_names[0]}_x_{exp_names[1]}_{target_n}_FWE05.png",
                                                  nii_folder=f"Results/Contrast/{cat}/{exp_names[0]}_x_{exp_names[1]}_{target_n}_FWE05.nii")

    print(f'{exp_names[0]} x {exp_names[1]} - {cat} contrast done!')