import os
from os.path import isfile
import nibabel as nb
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
import pickle
from compute_ale import compute_ale
from compile_studies import compile_studies
from contribution import contribution
from nilearn import plotting
from scipy import ndimage
from kernel import kernel_conv

def par_perm(s1,s2,vp,benchmark,shape,direction):
    arr = np.zeros(shape)
    sr = np.arange(len(s1)+len(s2))
    np.random.shuffle(sr)
    if direction == "positive":
        draw = (1-np.prod(vp[sr[:len(s1)],:], axis=0)) - (1-np.prod(vp[sr[len(s1):],:], axis=0))
    elif direction == "negative":
        draw = (1-np.prod(vp[sr[len(s1):],:], axis=0)) - (1-np.prod(vp[sr[:len(s1)],:], axis=0))
    else:
        print("Please specify direction of contrast.")
    arr[draw >= benchmark] = 1
    return arr

def compute_contrast(experiment1, study1, experiment2, study2):
    
    repeats=25000
    u = 0.05
    eps=np.finfo(float).eps

    cwd = os.getcwd()
    mask_folder = cwd + "/MaskenEtc/"
    try:
        os.makedirs(cwd + "/ALE/Contrasts/Images")
        os.makedirs(cwd + "/ALE/Conjunctions/Images")
    except:
        pass

    template = nb.load(mask_folder + "Grey10.nii")
    template_data = template.get_fdata()
    template_shape = template_data.shape
    pad_tmp_shape = [value+30 for value in template_shape]
    bg_img = nb.load(mask_folder + "MNI152.nii")
    
    s1 = list(range(experiment1.shape[0]))
    s2 = list(range(experiment2.shape[0]))


    if isfile("{}/ALE/Contrasts/{}--{}_P95.nii".format(cwd, study1, study2)):
        print("Loading contrast.")
        contrast_arr = nb.load("{}/ALE/Contrasts/{}--{}_P95.nii".format(cwd,study1,study2)).get_fdata()
    else:
        print("Computing positive contrast.")
        fx1 = nb.load("{}/ALE/Results/{}_cFWE05.nii".format(cwd,study1)).get_fdata()
        ind = np.where(fx1 > 0)
        if ind[0].size > 0:
            z1 = fx1[fx1 > 0]
            vp = np.zeros((len(s1)+len(s2), z1.size))

            for i in s1:
                data = kernel_conv(i, experiment1, pad_tmp_shape)
                vp[i,:] = data[tuple(ind)]

            for i in s2:
                data = kernel_conv(i, experiment2, pad_tmp_shape)
                vp[i+len(s1),:] = data[tuple(ind)]

            vp = 1-vp
            benchmark = (1-np.prod(vp[:len(s1),:], axis=0)) - (1-np.prod(vp[len(s1):,:], axis=0))

            print("Randomising (positive).")
            result = Parallel(n_jobs=-1,backend="threading")(delayed(par_perm)(s1, s2, vp, benchmark, ind[0].shape, direction="positive") for i in range(repeats))
            a = np.sum(result, axis=0)
            a = norm.ppf(1-(a/repeats))
            a[np.logical_and(np.isinf(a), a>0)] = norm.ppf(1-eps)

            z1 = np.minimum(z1, a)
            ind1 = np.argwhere(z1 > norm.ppf(1-u))
            idx_list1 = np.array([[ind[0][idx][0], ind[1][idx][0], ind[2][idx][0]] for idx in ind1]).T
            z1 = z1[z1 > norm.ppf(1-u)]

        else:
            print("{}: No significant indices!".format(study1))
            ind1, z1 = [], []


        print("Computing negative contrast.")
        fx2 = nb.load("{}/ALE/Results/{}_cFWE05.nii".format(cwd,study2)).get_fdata()
        ind = np.where(fx2 > 0)
        if ind[0].size > 0:
            z2 = fx2[fx2 > 0]
            vp = np.zeros((len(s1)+len(s2), z2.size))

            for i in s1:
                data = kernel_conv(i, experiment1, pad_tmp_shape)
                vp[i,:] = data[tuple(ind)]

            for i in s2:
                data = kernel_conv(i, experiment2, pad_tmp_shape)
                vp[i+len(s1),:] = data[tuple(ind)]

            vp = 1-vp
            benchmark = (1-np.prod(vp[len(s1):,:], axis=0)) - (1-np.prod(vp[:len(s1),:], axis=0))

            print("Randomising (negative).")
            result = Parallel(n_jobs=-1,backend="threading")(delayed(par_perm)(s1, s2, vp, benchmark, ind[0].shape, direction="negative") for i in range(repeats))
            a = np.sum(result, axis=0)
            a = norm.ppf(1-(a/repeats))
            a[np.logical_and(np.isinf(a), a>0)] = norm.ppf(1-eps)

            z2 = np.minimum(z2, a)
            ind2 = np.argwhere(z2 > norm.ppf(1-u))
            idx_list2 = np.array([[ind[0][idx][0], ind[1][idx][0], ind[2][idx][0]] for idx in ind2]).T
            z2 = z2[z2 > norm.ppf(1-u)]

        else:
            print("{}: No significant indices!".format(study2))
            ind2, z2 = [], []

        print("Inference and printing.")
        contrast_arr = np.zeros(template_shape)
        contrast_arr[tuple(idx_list1)] = z1
        contrast_arr[tuple(idx_list2)] = -z2
        contrast_img = nb.Nifti1Image(contrast_arr, template.affine)
        plotting.plot_stat_map(contrast_img, bg_img=bg_img, output_file="{}/ALE/Contrasts/Images/{}--{}_P95.png".format(cwd,study1,study2))
        nb.save(contrast_img, "{}/ALE/Contrasts/{}--{}_P95.nii".format(cwd,study1,study2))

    if isfile("{}/ALE/Conjunctions/{}_AND_{}_P95.nii".format(cwd,study1,study2)):
        print("Loading conjunction.")
        conj_arr = nb.load("{}/ALE/Conjunctions/{}_AND_{}_P95.nii".format(cwd,study1,study2)).get_fdata()
    else:
        print("Computing conjunction.")
        min_arr = np.minimum(fx1, fx2)
        conj_coords = np.where(min_arr > 0)
        cluster_arr = np.zeros(template_shape)
        cluster_arr[tuple(conj_coords)] = 1
        label_arr, cluster_count = ndimage.label(cluster_arr)
        labels, count = np.unique(label_arr[label_arr>0], return_counts=True)

        conj_arr = np.zeros(template_shape)
        for i in labels:
            if count[i-1] > 50:
                conj_arr[label_arr == i] = min_arr[label_arr == i]

        conj_img = nb.Nifti1Image(conj_arr, template.affine)
        plotting.plot_stat_map(contrast_img, bg_img=bg_img, output_file="{}/ALE/Conjunctions/Images/{}_AND_{}_P95.png".format(cwd,study1,study2))
        nb.save(conj_img, "{}/ALE/Conjunctions/{}_AND_{}_P95.nii".format(cwd,study1,study2))