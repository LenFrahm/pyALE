import os
from os.path import isfile
import nibabel as nb
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from nilearn import plotting
from scipy import ndimage
from kernel import kernel_conv
from main_effect import plot_and_save, compute_ma
# importing brain template information
from template import shape, pad_shape, prior, affine

REPEATS = 25000 #number of permutations
U = 0.05 # significance level by which to threshold z-values
EPS = np.finfo(float).eps # float precision


def compute_ale_diff(exp_df1, exp_df2, s1, s2, ind):   
    peaks1 = np.array([exp_df1.XYZ[i].T[:,:3] for i in s1], dtype=object)
    peaks2 = np.array([exp_df2.XYZ[i].T[:,:3] for i in s2], dtype=object)

    ma1 = compute_ma(s1, peaks1, exp_df1.Kernels)[:,ind[0],ind[1],ind[2]]
    ma2 = compute_ma(s2, peaks2, exp_df2.Kernels)[:,ind[0],ind[1],ind[2]]
    ma = np.vstack((ma1,ma2))

    ma = 1-ma   # MA map for each study in both experiments
    # Actual observed difference in ALE values between the two meta analysis
    ale_diff = (1-np.prod(ma[:len(s1)], axis=0)) - (1-np.prod(ma[len(s1):], axis=0))
    
    return ma, ale_diff


def par_perm_diff(s1,s2,ma,ale_diff):
    # make list with range of values with amount of studies in both experiments together
    sr = np.arange(len(s1)+len(s2))
    np.random.shuffle(sr)
    # calculate ale difference for this permutation
    perm_diff = (1-np.prod(ma[sr[:len(s1)]], axis=0)) - (1-np.prod(ma[sr[len(s1):]], axis=0))
    # set voxels where perm_diff is bigger than the actually observed diff to 1
    null_diff = np.zeros(ale_diff.shape)
    null_diff[perm_diff >= ale_diff] = 1
    
    return null_diff


def compute_sig_diff(z, null_diff):
    # sum all times the permutation was bigger than the actual difference and divide it by the number of repititions -> ~0.5 if voxel shows no siginifcant difference
    a = np.sum(null_diff, axis=0)/ REPEATS
    a = norm.ppf(1-a) # z-value
    a[np.logical_and(np.isinf(a), a>0)] = norm.ppf(1-EPS)

    z = np.minimum(z, a) # for values where the actual difference is consistently higher than the null distribution the minimum will be z and ->
    sig_idxs = np.argwhere(z > norm.ppf(1-U)).T # will most likely be above the threshold of p = 0.05 or z ~ 1.65
    z = z[sig_idxs]
    
    return z, sig_idxs


def compute_conjunction(fx1, fx2):

    min_arr = np.minimum(fx1, fx2)
    conj_coords = np.where(min_arr > 0)
    cluster_arr = np.zeros(shape)
    cluster_arr[tuple(conj_coords)] = 1
    label_arr, cluster_count = ndimage.label(cluster_arr)
    labels, count = np.unique(label_arr[label_arr>0], return_counts=True)

    conj_arr = np.zeros(shape)
    counter = 0
    for i in labels:
        if count[i-1] > 50:
            conj_arr[label_arr == i] = min_arr[label_arr == i]
            counter += 1
    if counter == 0:
        return None
    
    return conj_arr


def compute_contrast(exp_df1, exp_name1, exp_df2, exp_name2):    
    # Create necessary folder structure
    cwd = os.getcwd()
    mask_folder = f"{cwd}/MaskenEtc/"
    try:
        os.makedirs(f"{cwd}/ALE/Unbalanced/Contrasts/Images")
        os.makedirs(f"{cwd}/ALE/Unbalanced/Conjunctions/Images")
    except:
        pass
    
    # Declare variables for future calculations
    # simple lists containing numbers 0 to number of studies -1 for iteration over studies
    s1 = list(range(exp_df1.shape[0]))
    s2 = list(range(exp_df2.shape[0]))
    
    fx1 = nb.load(f"{cwd}/ALE/MainEffect/Results/{exp_name1}_cFWE05.nii").get_fdata()
    fx2 = nb.load(f"{cwd}/ALE/MainEffect/Results/{exp_name2}_cFWE05.nii").get_fdata()

    # Check if contrast has already been calculated
    if isfile(f"{cwd}/ALE/Unbalanced/Contrasts/{exp_name1}--{exp_name2}_P95.nii"):
        print(f"{exp_name1} x {exp_name2} - Loading contrast.")
        contrast_arr = nb.load(f"{cwd}/ALE/Unbalanced/Contrasts/{exp_name1}--{exp_name2}_P95.nii").get_fdata()
    else:
        print(f"{exp_name1} x {exp_name2} - Computing positive contrast.")
        ind = np.where(fx1 > 0)
        if ind[0].size > 0:
            z1 = fx1[fx1 > 0]
            ma, ale_diff = compute_ale_diff(exp_df1, exp_df2, s1, s2, ind)
            # estimate null distribution of difference values if studies would be randomly assigned to either meta analysis
            null_diff = Parallel(n_jobs=-1,backend="threading")(delayed(par_perm_diff)(s1, s2, ma, ale_diff) for i in range(REPEATS))
            z1, sig_idxs1 = compute_sig_diff(z1, null_diff)
            sig_idxs1 = np.array(ind)[:,sig_idxs1].squeeze()

        else:
            print(f"{exp_name1}: No significant indices!")
            z1, sig_idxs1 = [], []


        print(f"{exp_name1} x {exp_name2} - Computing negative contrast.")
        ind = np.where(fx2 > 0)
        if ind[0].size > 0:
            z2 = fx2[fx2 > 0]
            # same calculation as above, but this time the studies are flipped in parameter position
            ma, ale_diff = compute_ale_diff(exp_df2, exp_df1, s2, s1, ind)
            null_diff = Parallel(n_jobs=-1,backend="threading")(delayed(par_perm_diff)(s2, s1, ma, ale_diff) for i in range(REPEATS))
            z2, sig_idxs2 = compute_sig_diff(z2, null_diff)
            sig_idxs2 = np.array(ind)[:,sig_idxs2].squeeze()
            

        else:
            print(f"{exp_name2}: No significant indices!")
            z2, sig_idxs2 = [], []

        print(f"{exp_name1} x {exp_name2} - Inference and printing.")
        contrast_arr = np.zeros(shape)
        contrast_arr[tuple(sig_idxs1)] = z1
        contrast_arr[tuple(sig_idxs2)] = -z2
        contrast_arr = plot_and_save(contrast_arr, img_folder=f"{cwd}/ALE/Unbalanced/Contrasts/Images/{exp_name1}--{exp_name2}_P95.png",
                                                   nii_folder=f"{cwd}/ALE/Unbalanced/Contrasts/{exp_name1}--{exp_name2}_P95.nii")
    
    #Check if conjunction has already been calculated
    if isfile(f"{cwd}/ALE/Unbalanced/Conjunctions/{exp_name1}_AND_{exp_name2}_P95.nii"):
        print(f"{exp_name1} & {exp_name2} - Loading conjunction.")
        conj_arr = nb.load(f"{cwd}/ALE/Unbalanced/Conjunctions/{exp_name1}_AND_{exp_name2}_P95.nii").get_fdata()
    else:
        print(f"{exp_name1} & {exp_name2} - Computing conjunction.")
        conj_arr = compute_conjunction(fx1, fx2) 
        if conj_arr is not None:
            conj_arr = plot_and_save(conj_arr, img_folder=f"{cwd}/ALE/Unbalanced/Conjunctions/Images/{exp_name1}_AND_{exp_name2}_P95.png",
                                               nii_folder=f"{cwd}/ALE/Unbalanced/Conjunctions/{exp_name1}_AND_{exp_name2}_P95.nii")