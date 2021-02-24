import os
from os.path import isfile
import nibabel as nb
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
import pickle
from nilearn import plotting
from scipy import ndimage
from kernel import kernel_conv
from compile_studies import compile_studies
from template import prior, affine, shape, pad_shape, sample_space
from main_effect import *

THRESH = 0.001
SAMPLE_N = 2500
REPEATS = 1000
REPLICATES = 500


def create_samples(s0, sample_space, sample_n, target_n):
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


def compute_sub_ale(sample, ma, hx, bin_centers, cut_cluster):
    ale_null = compute_ale_null(sample, hx, bin_centers)
    ale = compute_ale(ma[sample])
    z = compute_z(ale, ale_null)
    z, max_cluster = compute_cluster(z, sample_space=None, cut_cluster=cut_cluster)
    z[z > 0] = 1
    return z


def compute_diff(s,ma_maps):
    ale = np.zeros((2,prior.sum()))
    for xi in (0,1):
        sx = np.random.permutation(s[xi])
        sx = sx[:target_n]
        ale[xi,:] = compute_ale(ma_maps[xi][:,prior][sx,:])
    r_diff = ale[0,:] - ale[1,:]
    return r_diff


def compute_noise_diff(s, sample_space, exp_df):
    noise_ma = []
    for xi in (0,1):
        noise_peaks = np.array([prior_idx[np.random.randint(0, prior_idx.shape[0], exp_df[xi].Peaks[i]), :] for i in s[xi]], dtype=object)
        noise_ma.append(compute_ma(s[xi], noise_peaks, exp_df[xi].Kernels))

    p_diff = Parallel(prefer="threads", n_jobs=-1, verbose=1)(delayed(compute_diff)(s,noise_ma) for i in range(REPEATS))
    p_diff_mean = np.mean(p_diff, axis=0)
    min_diff, max_diff = np.min(p_diff_mean), np.max(p_diff_mean)
    return min_diff, max_diff


def balanced_contrast(exp_name, exp_idxs, exp_df, s, target_n):
    
    cwd = os.getcwd()
    try:
        os.makedirs(f"{cwd}/ALE/Balanced/Contrasts/NullDistributions")
        os.makedirs(f"{cwd}/ALE/Balanced/Contrasts/SubALE")
        os.makedirs(f"{cwd}/ALE/Balanced/Contrasts/Images/SubALE")
        os.makedirs(f"{cwd}/ALE/Balanced/Conjunctions/Images")
    except:
        pass

    real_ma = []
    cut_clusters = []
    sub_ale_maps = []
    for xi in (0,1):
    # highest possible ale value if every study had a peak at the same location.
        mb = 1
        for i in s[xi]:
            mb = mb*(1-np.max(exp_df[xi].at[i, 'Kernels']))

        # define bins for histogram
        bin_edges = np.arange(0.00005,1-mb+0.001,0.0001)
        bin_centers = np.arange(0,1-mb+0.001,0.0001)

        if isfile(f"{cwd}/ALE/Balanced/Contrasts/NullDistributions/{exp_name[xi]}_ccut_{target_n}.pickle"):
            print(f'{exp_name[xi]} - loading cluster null')
            with open(f"{cwd}/ALE/Balanced/Contrasts/NullDistributions/{exp_name[xi]}_ccut_{target_n}.pickle", 'rb') as f:
                cut_cluster = pickle.load(f)   
                cut_clusters.append(cut_cluster)
        else:
            print(f'{exp_name[xi]} - computing cluster null')
            # Cluster Null
            max_ale, max_cluster = zip(*Parallel(n_jobs=8, verbose=1)(delayed(compute_noise_max)(s0 = s[xi],
                                                                                      sample_space = sample_space,
                                                                                      num_peaks = exp_df[xi].Peaks,
                                                                                      kernels = exp_df[xi].Kernels,
                                                                                      bin_centers=bin_centers,
                                                                                      bin_edges=bin_edges,
                                                                                      target_n=17) for i in range(1000)))
            cut_cluster = np.percentile(max_cluster, 95)
            cut_clusters.append(cut_cluster)
            with open(f"{cwd}/ALE/Balanced/Contrasts/NullDistributions/{exp_name[xi]}_ccut_{target_n}.pickle", "wb") as f:
                pickle.dump(cut_cluster, f)

        if isfile(f"{cwd}/ALE/Balanced/Contrasts/SubALE/{exp_name[xi]}_{target_n}.nii"):
            print(f'{exp_name[xi]} - loading sub ALE')
            sub_ale_maps.append(nb.load(f"ALE/Balanced/Contrasts/SubALE/{exp_name[xi]}_{target_n}.nii").get_fdata())
        else:
            print(f'{exp_name[xi]} - computing sub ALE')
            # SubALE
            peaks = np.array([exp_df[xi].XYZ[i].T[:,:3] for i in s[xi]], dtype=object)
            ma = compute_ma(s[xi], peaks, exp_df[xi].Kernels)
            real_ma.append(ma)
            hx = compute_hx(s[xi], ma, bin_edges)

            samples = create_samples(s[xi], sample_space, SAMPLE_N, target_n)

            sub_ale = Parallel(n_jobs=8, verbose=2)(delayed(compute_sub_ale)(samples[i], ma, hx, bin_centers, cut_cluster) for i in range(samples.shape[0]))
            sub_ale = np.mean(sub_ale, axis=0)
            sub_ale = plot_and_save(sub_ale, img_folder=f"ALE/Balanced/Contrasts/Images/SubALE/{exp_name[xi]}_{target_n}.png",
                                           nii_folder=f"ALE/Balanced/Contrasts/SubALE/{exp_name[xi]}_{target_n}.nii")
            sub_ale_maps.append(sub_ale)


    sub_ale1, sub_ale2 = sub_ale_maps

    if isfile(f"ALE/Balanced/Conjunctions/{exp_name[0]}_AND_{exp_name[1]}_{target_n}.nii"):
        print(f'{exp_name[0]} x {exp_name[1]} - loading conjunction')
        pass
    else:
        print(f'{exp_name[0]} x {exp_name[1]} - computing conjunction')
        conjunction = np.minimum(sub_ale1, sub_ale2)
        conjunction = plot_and_save(conjunction, img_folder=f"ALE/Balanced/Conjunctions/Images/{exp_name[0]}_AND_{exp_name[1]}_{target_n}.png",
                                                 nii_folder=f"ALE/Balanced/Conjunctions/{exp_name[0]}_AND_{exp_name[1]}_{target_n}.nii")

    if isfile(f"ALE/Balanced/Contrasts/NullDistributions/{exp_name[0]}_x_{exp_name[0]}_{target_n}.pickle"):
        print(f'{exp_name[0]} x {exp_name[1]} - loading actual diff and noise extremes')
        with open(f"ALE/Balanced/Contrasts/NullDistributions/{exp_name[0]}_x_{exp_name[0]}_{target_n}.pickle", 'rb') as f:
            r_diff, prior, min_diff, max_diff = pickle.load(f)
    else:
        print(f'{exp_name[0]} x {exp_name[1]} - computing actual diff and noise extremes')
        prior = (sub_ale1 + sub_ale2) > 0.05
        prior_idx = np.argwhere(prior > 0)

        r_diff = Parallel(n_jobs=8, verbose=1)(delayed(compute_diff)(s,real_ma) for i in range(REPEATS))
        r_diff = np.mean(r_diff, axis=0)

        min_diff, max_diff = zip(*Parallel(n_jobs=8, verbose=1)(delayed(compute_noise_diff)(s, sample_space, exp_df) for i in range(REPLICATES)))

        pickle_object = (r_diff, mask, min_diff, max_diff)
        with open(f"{cwd}/ALE/Balanced/Contrasts/NullDistributions/{exp_name[0]}_x_{exp_name[0]}_{target_n}.pickle", "wb") as f:
            pickle.dump(pickle_object, f)

    if isfile(f"ALE/Balanced/Contrasts/{exp_name[0]}_x_{exp_name[1]}_{target_n}_FWE05.nii"):
        print(f'{exp_name[0]} x {exp_name[1]} - balanced contrast finished')
        pass
    else:
        print(f'{exp_name[0]} x {exp_name[1]} - computing significant contrast')
        min_cut, max_cut = np.percentile(min_diff, 2.5), np.percentile(max_diff, 97.5)  

        sig_diff = r_diff * np.logical_or(r_diff < min_cut, r_diff > max_cut)
        sig_diff[sig_diff > 0] = np.array([-1 * norm.ppf((sum(max_diff >= diff)+1) / (REPLICATES+1)) for diff in sig_diff[sig_diff > 0]])
        sig_diff[sig_diff < 0] = np.array([norm.ppf((sum(max_diff <= diff)+1) / (REPLICATES+1)) for diff in sig_diff[sig_diff < 0]])

        brain_sig_diff = np.zeros(shape)
        brain_sig_diff[prior] = sig_diff

        sig_diff2 = plot_and_save(brain_sig_diff, img_folder=f"ALE/Balanced/Contrasts/Images/{exp_name[0]}_x_{exp_name[1]}_{target_n}_FWE05.png",
                                                  nii_folder=f"ALE/Balanced/Contrasts/{exp_name[0]}_x_{exp_name[1]}_{target_n}_FWE05.nii")