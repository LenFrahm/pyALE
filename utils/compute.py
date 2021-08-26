import os
import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import norm
from nilearn import plotting
from joblib import Parallel, delayed
from utils.tfce_par import tfce_par
from utils.template import shape, pad_shape, prior, affine, sample_space
from utils.kernel import kernel_conv

EPS =  np.finfo(float).eps
cwd = os.getcwd()

""" Main Effect Computations """


def illustrate_foci(peaks):    
    foci_arr = np.zeros(shape)    
    # Load all peaks associated with study
    peaks = np.concatenate(peaks)
    #Set all points in foci_arr that are peaks for the study to 1
    foci_arr[tuple(peaks.T)] += 1
    
    return foci_arr


def compute_ma(peaks, kernels):
    ma = np.zeros((len(kernels), shape[0], shape[1], shape[2]))
    for i, kernel in enumerate(kernels):
        ma[i, :] = kernel_conv(peaks = peaks[i], 
                               kernel = kernel)
        
    return ma


def compute_hx(ma, bin_edges):
    hx = np.zeros((ma.shape[0], len(bin_edges)))
    for i in range(ma.shape[0]):
        data = ma[i, :]
        bin_idxs, counts = np.unique(np.digitize(data[prior], bin_edges),return_counts=True)
        hx[i,bin_idxs] = counts
    return hx


def compute_ale(ma):
    return 1-np.prod(1-ma, axis=0)


def compute_hx_conv(hx, bin_centers, step):    
    ale_hist = hx[0,:]
    for x in range(1,hx.shape[0]):
        v1 = ale_hist
        # save bins, which there are entries in the combined hist
        da1 = np.where(v1 > 0)[0]
        # normalize combined hist to sum to 1
        v1 = ale_hist/np.sum(v1)
        
        v2 = hx[x,:]
        # save bins, which there are entries in the study hist
        da2 = np.where(v2 > 0)[0]
        # normalize study hist to sum to 1
        v2 = hx[x,:]/np.sum(v2)
        ale_hist = np.zeros((len(bin_centers),))
        #iterate over bins, which contain values
        for i in range(len(da2)):
            p = v2[da2[i]]*v1[da1]
            score = 1-(1-bin_centers[da2[i]])*(1-bin_centers[da1])
            ale_bin = np.round(score*step).astype(int)
            ale_hist[ale_bin] = np.add(ale_hist[ale_bin], p)
    last_used = np.where(ale_hist>0)[0][-1]
    hx_conv = np.flip(np.cumsum(np.flip(ale_hist[:last_used+1])))
    
    return hx_conv


def compute_z(ale, hx_conv, step):    
    # computing the corresponding histogram bin for each ale value
    ale_step = np.round(ale*step).astype(int)
    # replacing histogram bin number with corresponding histogram value (= p-value)
    p = np.array([hx_conv[i] for i in ale_step])
    p[p < EPS] = EPS
    # calculate z-values by plugging 1-p into a probability density function
    z = norm.ppf(1-p)
    
    return z


def compute_tfce(z):
    
    if z.shape != shape:
        tmp = np.zeros(shape)
        tmp[tuple(sample_space)] = z
        z = tmp
    
    lc = np.min(sample_space, axis=1)
    uc = np.max(sample_space, axis=1)  
    z = z[lc[0]:uc[0],lc[1]:uc[1],lc[2]:uc[2]]
    
    delta_t = np.max(z)/100
    
    tmp = np.zeros(z.shape)
    # calculate tfce values using the parallelized function
    vals, masks = zip(*Parallel(n_jobs=4)
                     (delayed(tfce_par)(invol=z, h=h, dh=delta_t) for h in np.arange(0, np.max(z), delta_t)))
    # Parallelization makes it necessary to integrate the results afterwards
    # Each repition creats it's own mask and an amount of values corresponding to that mask
    for i in range(len(vals)):
        tmp[masks[i]] += vals[i]
        
    tfce = np.zeros(shape)
    tfce[lc[0]:uc[0],lc[1]:uc[1],lc[2]:uc[2]] = tmp
    
    return tfce


def compute_cluster(z, thresh, sample_space=None, cut_cluster=None):    
    # disregard all voxels that feature a z-value of lower than some threshold (approx. 3 standard deviations aways from the mean)
    # this serves as a preliminary thresholding
    sig_arr = np.zeros(shape)
    sig_arr[z > norm.ppf(1-thresh)] = 1
    # find clusters of significant z-values
    labels, cluster_count = ndimage.label(sig_arr)
    # save number of voxels in biggest cluster
    max_clust = np.max(np.bincount(labels[labels>0]))
    if cut_cluster is not None:
        # check significance of cluster against the 95th percentile of the null distribution cluster size
        sig_clust = np.where(np.bincount(labels[labels > 0]) > cut_cluster)[0]
        # z-value array that only features values for voxels that belong to significant clusters
        z = z*np.isin(labels, sig_clust)
        return z, max_clust
    
    return max_clust


def compute_null_ale(s0, sample_space, num_peaks, kernels):   
    null_peaks = np.array([sample_space[:,np.random.randint(0,sample_space.shape[1], num_peaks[i])].T for i in s0], dtype=object)
    null_ma = compute_ma(null_peaks, kernels)
    null_ale = compute_ale(null_ma)
    
    return null_ma, null_ale
    
    
def compute_null_cutoffs(s0, sample_space, num_peaks, kernels, step=10000, thresh=0.001, target_n=None,
                          hx_conv=None, bin_edges=None, bin_centers=None, tfce=None):
    if target_n:
        s0 = np.random.permutation(s0)
        s0 = s0[:target_n]
    # compute ALE values based on random peak locations sampled from a give sample_space
    # sample space could be all grey matter or only foci reported in brainmap
    null_ma, null_ale = compute_null_ale(s0, sample_space, num_peaks, kernels)
    # Peak ALE threshold
    null_max_ale = np.max(null_ale)
    if hx_conv is None:
        s0 = list(range(len(s0)))
        null_hx = compute_hx(null_ma, bin_edges)
        hx_conv = compute_hx_conv(null_hx, bin_centers, step)
    null_z = compute_z(null_ale, hx_conv, step)
    # Cluster level threshold
    null_max_cluster = compute_cluster(null_z, thresh, sample_space)
    if tfce:
        tfce = compute_tfce(null_z)
        # TFCE threshold
        null_max_tfce = np.max(tfce)
        return null_max_ale, null_max_cluster, null_max_tfce
        
    return null_max_ale, null_max_cluster

""" CV/Subsampling ALE Computations """


def create_samples(s0, sample_n, target_n):
    samples = np.zeros((sample_n, target_n))
    s_perm = s0.copy()
    for i in range(sample_n):
        np.random.shuffle(s_perm)
        samples[i,:] = np.sort(s_perm[:target_n])
    samples = np.unique(samples, axis=0)
    unique_samples = samples.shape[0]
    i = 0
    while unique_samples < sample_n:
        np.random.shuffle(s_perm)
        samples = np.vstack((samples, np.sort(s_perm[:target_n])))
        samples = np.unique(samples, axis=0)
        unique_samples = samples.shape[0]
        i += 1
        if i == sample_n:
            return samples.astype(int)
    return samples.astype(int)


def compute_sub_ale(sample, ma, hx, bin_centers, cut_cluster, step=10000, thresh=0.001):
    hx_conv = compute_hx_conv(sample, hx, bin_centers, step)
    ale = compute_ale(ma[sample])
    z = compute_z(ale, hx_conv, step)
    z, max_cluster = compute_cluster(z, thresh, sample_space=None, cut_cluster=cut_cluster)
    z[z > 0] = 1
    return z


""" New Contrast Computations """


def compute_ale_diff(s, ma_maps, prior, target_n=None):
    ale = np.zeros((2,prior.sum()))      
    for xi in (0,1):
        if target_n:  
            s_perm = np.random.permutation(s[xi])
            s_perm = s_perm[:target_n]
            ale[xi,:] = compute_ale(ma_maps[xi][:,prior][s_perm,:])
        else:
            ale[xi,:] = compute_ale(ma_maps[xi][:,prior])
    r_diff = ale[0,:] - ale[1,:]
    return r_diff


def compute_null_diff(s, prior, exp_dfs, target_n=None, diff_repeats=1000):
    prior_idxs = np.argwhere(prior > 0)
    null_ma = []
    for xi in (0,1):
        null_peaks = np.array([prior_idxs[np.random.randint(0, prior_idxs.shape[0], exp_dfs[xi].Peaks[i]), :] for i in s[xi]], dtype=object)
        null_ma.append(compute_ma(s[xi], null_peaks, exp_dfs[xi].Kernels))
    
    if target_n:
        p_diff = Parallel(n_jobs=4, verbose=1)(delayed(compute_ale_diff)(s, null_ma, prior, target_n) for i in range(diff_repeats))
        p_diff = np.mean(p_diff, axis=0)
    else:
        p_diff = compute_diff(s, null_ma, prior)
        
    min_diff, max_diff = np.min(p_diff), np.max(p_diff)
    return min_diff, max_diff


""" Legacy Contrast Computations"""


def compute_perm_diff(s,masked_ma):
    # make list with range of values with amount of studies in both experiments together
    sr = np.arange(len(s[0])+len(s[1]))
    sr = np.random.permutation(sr)
    # calculate ale difference for this permutation
    perm_diff = (1-np.prod(masked_ma[sr[:len(s[0])]], axis=0)) - (1-np.prod(masked_ma[sr[len(s[0]):]], axis=0))
    return perm_diff

def compute_sig_diff(fx, mask, ale_diff, perm_diff, null_repeats, diff_thresh):
    n_bigger = [np.sum([diff[i] > ale_diff[i] for diff in perm_diff]) for i in range(mask.sum())]
    prob_bigger = np.array([x / null_repeats for x in n_bigger])

    z_null = norm.ppf(1-prob_bigger) # z-value
    z_null[np.logical_and(np.isinf(z_null), z_null>0)] = norm.ppf(1-EPS)
    z = np.minimum(fx[mask], z_null) # for values where the actual difference is consistently higher than the null distribution the minimum will be z and ->
    sig_idxs = np.argwhere(z > norm.ppf(1-diff_thresh)).T # will most likely be above the threshold of p = 0.05 or z ~ 1.65
    z = z[sig_idxs]
    sig_idxs = np.argwhere(mask == True)[sig_idxs].squeeze().T
    return z, sig_idxs


""" Plot Utils """


def plot_and_save(arr, img_folder=None, nii_folder=None):
    # Function that takes brain array and transforms it to NIFTI1 format
    # Saves it both as a statmap png and as a Nifti file
    nii_img = nb.Nifti1Image(arr, affine)
    if img_folder:
        plotting.plot_stat_map(nii_img, output_file=img_folder)
    if nii_folder:
        nb.save(nii_img, nii_folder)
    
    return arr

