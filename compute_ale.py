import os
from os.path import isfile
import pandas as pd
import numpy as np
from scipy.stats import norm
from nilearn import plotting
import nibabel as nb
import math as m
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from compile_studies import compile_studies
from tfce import tfce_par, tfce
from scipy.io import loadmat
from simulate_noise import simulate_noise
from scipy import ndimage
from joblib import Parallel, delayed

def compute_ale(s_index, experiments, study, noise_repeat):
    
    cwd = os.getcwd()
    mask_folder = cwd + "/MaskenEtc/"
    
    folders_req = ['Volumes', 'NullDistributions', 'VolumesZ', 'VolumesTFCE', 'Results', 'Images', 'Foci']
    folders_req_imgs = ['Foci', 'ALE', 'TFCE']
    try:
        os.mkdir('ALE')
        for folder in folders_req:
            os.mkdir('ALE/' + folder)
            if folder == 'Images':
                for sub_folder in folders_req_imgs:
                    os.mkdir('ALE/Images/' + sub_folder)
    except FileExistsError:
        pass

    s0 = list(range(len(s_index)))
    
    template = nb.load(mask_folder + "Grey10.nii")
    template_data = template.get_fdata()
    template_shape = template_data.shape
    pad_tmp_shape = [value+30 for value in template_shape]
    bg_img = nb.load(mask_folder + "MNI152.nii")

    prior = np.zeros(template_shape, dtype=bool)
    prior[template_data > 0.1] = 1
    
    
    
    uc = 0.001
    eps = np.finfo(float).eps
    
    mb = 1
    for i in s0:
        mb = mb*(1-np.max(experiments.at[i, 'Kernel']))
    bin_edge = np.arange(0.00005,1-mb+0.001,0.0001)
    bin_center = np.arange(0,1-mb+0.001,0.0001)
    step = 1/0.0001

    if isfile(cwd + '/ALE/Foci/' + study + '.nii'):
        print('{} - loading Foci'.format(study))
        foci_arr = nb.load(cwd + '/ALE/Foci/' + study + '.nii').get_fdata()
    else:
        print('{} - illustrate Foci'.format(study))

        foci_arr = np.zeros(template_shape)
        nested_list = [experiments.XYZ[i].T[:,:3].tolist() for i in s0]
        flat_list = np.array([item for sublist in nested_list for item in sublist])
        foci_arr[tuple(flat_list.T)] += 1
        foci_arr[~prior] = np.nan 
        foci_img = nb.Nifti1Image(foci_arr, template.affine)
        plotting.plot_stat_map(foci_img, bg_img=bg_img, output_file="ALE/Images/Foci/" + study + ".png")
        nb.save(foci_img, cwd + '/ALE/Foci/' + study + '.nii')

        foci_arr = np.nan_to_num(foci_arr)


    if isfile(cwd + '/ALE/NullDistributions/' + study + '.pickle'):
        print('{} - loading ALE'.format(study))
        print('{} - loading null PDF'.format(study))
        ale = nb.load(cwd + '/ALE/Volumes/' + study + '.nii').get_fdata()
        ale = np.nan_to_num(ale)
        with open(cwd + '/ALE/NullDistributions/' + study + '.pickle', 'rb') as f:
                ale_hist, last_used, c_null = pickle.load(f)
    else:
        print('{} - computing ALE'.format(study))

        ale = np.ones(template_shape)
        hx = np.zeros((len(s0),len(bin_edge)))
        for i in s0:
            data = np.zeros(pad_tmp_shape)
            for ii in range(experiments.at[i, 'Peaks']):
                coords = experiments.XYZ[i].T[:,:3][ii]
                x_range = (coords[0],coords[0]+31)
                y_range = (coords[1],coords[1]+31)
                z_range = (coords[2],coords[2]+31)
                data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] = \
                np.maximum(data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]],
                           experiments.at[i, 'Kernel'])
            data = data[15:data.shape[0]-15,15:data.shape[1]-15, 15:data.shape[2]-15]
            bin_idxs, counts = np.unique(np.digitize(data[prior], bin_edge),return_counts=True)
            hx[c,bin_idxs] = counts
            ale = np.multiply(ale, 1-data)

        ale = 1-ale
        ale[~prior] = np.nan

        #Save ALE scores in Nifti
        ale_img = nb.Nifti1Image(ale, template.affine)
        plotting.plot_stat_map(ale_img, bg_img=bg_img, output_file="ALE/Images/ALE/" + study + ".png")
        nb.save(ale_img, cwd + '/ALE/Volumes/' + study + '.nii')

        ale = np.nan_to_num(ale)

        print('{} - permutation-null PDF'.format(study))
        step = 1/np.mean(np.diff(bin_center))
        ale_hist = hx[0,:]
        for i in range(1,len(s0)):
            v1 = ale_hist
            v2 = hx[i,:]

            da1 = np.where(v1 > 0)[0]
            da2 = np.where(v2 > 0)[0]

            v1 = ale_hist/np.sum(v1)
            v2 = hx[i,:]/np.sum(v2)

            ale_hist = np.zeros((len(bin_center),))
            for i in range(len(da2)):
                p = v2[da2[i]]*v1[da1]
                score = 1-(1-bin_center[da2[i]])*(1-bin_center[da1])
                ale_bin = np.round(score*step).astype(int)
                ale_hist[ale_bin] = np.add(ale_hist[ale_bin], p)

        last_used = np.where(ale_hist>0)[0][-1]
        c_null = np.flip(np.cumsum(np.flip(ale_hist[:last_used+1])))

        pickle_object = (ale_hist, last_used, c_null)
        with open(cwd + '/ALE/NullDistributions/' + study + '.pickle', "wb") as f:
            pickle.dump(pickle_object, f)

    if isfile(cwd + '/ALE/VolumesTFCE/' + study + '.nii'):
        print('{} - loading p-values'.format(study))

        z = nb.load(cwd + '/ALE/VolumesZ/' + study + '.nii').get_fdata()
        z = np.nan_to_num(z)

        tfce_arr = nb.load(cwd + '/ALE/VolumesTFCE/' + study + '.nii').get_fdata()
        tfce_arr = np.nan_to_num(tfce_arr)
    else:
        print('{} - computing p-values'.format(study))

        ale_step = np.round(ale*step).astype(int)
        p = np.array([c_null[i] for i in ale_step])
        p[p < eps] = eps
        z = norm.ppf(1-p)

        tfce_arr = tfce(invol=z, voxel_dims=template.header.get_zooms())
        tfce_arr[~prior] = np.nan

        tfce_img = nb.Nifti1Image(tfce_arr, template.affine)
        plotting.plot_stat_map(tfce_img, bg_img=bg_img, output_file="ALE/Images/TFCE/" + study + ".png")
        nb.save(tfce_img, cwd + '/ALE/VolumesTFCE/' + study + '.nii')

        tfce_arr = np.nan_to_num(tfce_arr)

        z[~prior] = np.nan
        z_img = nb.Nifti1Image(z, template.affine)
        nb.save(z_img, cwd + '/ALE/VolumesZ/' + study + '.nii')

        z = np.nan_to_num(z)

    if isfile(cwd+"/ALE/NullDistributions/" + study + "_clustP.pickle"):
        print('{} - loading noise'.format(study))
        with open(cwd+"/ALE/NullDistributions/" + study + "_clustP.pickle", 'rb') as f:
            nm, nn, nt = pickle.load(f)
    else:
        print('{} - simulating noise'.format(study))
        num_peaks = experiments.loc[:,'Peaks']
        kernels = experiments.loc[:,'Kernel']

        permSpace5 = loadmat("MaskenEtc/permSpace5.mat")
        sample_space = permSpace5["allXYZ"]
        delta_t = np.max(z)/100

        nm, nn, nt = zip(*Parallel(n_jobs=3, verbose=5)(delayed(simulate_noise)(sample_space = sample_space,
                                                                         s0 = s0,
                                                                         num_peaks = num_peaks,
                                                                         kernels = kernels,
                                                                         c_null = c_null,
                                                                         tfce_params = [delta_t, 0.6, 2]) for i in range(noise_repeat)))


        simulation_pickle = (nm, nn, nt)
        with open(cwd+"/ALE/NullDistributions/" + study + "_clustP.pickle", "wb") as f:
            pickle.dump(simulation_pickle, f)

    if not isfile(cwd + '/ALE/Results/' + study + '_TFCE05.nii'):
        print('{} - inference and printing'.format(study))
        # voxel wise family wise error correction
        cut_max = np.percentile(nm, 95)

        ale = ale*(ale > cut_max)
        ale_img = nb.Nifti1Image(ale, template.affine)
        plotting.plot_stat_map(ale_img, bg_img=bg_img, output_file="ALE/Images/" + study + "_FWE05.png")
        nb.save(ale_img, cwd + '/ALE/Results/' + study + '_FWE05.nii')

        print("Min p-value for FWE:" + str(sum(nm>np.max(ale))/len(nn)))

        # cluster wise family wise error correction
        cut_clust = np.percentile(nn, 95)

        z[z < norm.ppf(1-uc)] = 0
        sig_coords = np.where(z > 0)
        sig_arr = np.zeros(template_shape)
        sig_arr[tuple(sig_coords)] = 1
        labels, cluster_count = ndimage.label(sig_arr)
        sig_clust = np.where(np.bincount(labels[labels > 0]) > cut_clust)[0]
        max_clust = np.max(np.bincount(labels[labels>0]))
        z = z*np.isin(labels, sig_clust)
        cfwe_img = nb.Nifti1Image(z, template.affine)
        plotting.plot_stat_map(cfwe_img, bg_img=bg_img, output_file="ALE/Images/" + study + "_cFWE05.png")
        nb.save(cfwe_img, cwd + '/ALE/Results/' + study + '_cFWE05.nii')

        print("Min p-value for cFWE:" + str(sum(nn>max_clust)/len(nm)))

        # tfce error correction
        cut_tfce = np.percentile(nt, 95)

        tfce_arr = tfce_arr*(tfce_arr > cut_tfce)
        tfce_img = nb.Nifti1Image(tfce_arr, template.affine)
        plotting.plot_stat_map(tfce_img, bg_img=bg_img, output_file="ALE/Images/" + study + "_TFCE05.png")
        nb.save(tfce_img, cwd + '/ALE/Results/' + study + '_TFCE05.nii')

        print("Min p-value for TFCE:" + str(sum(nt>np.max(tfce_arr))/len(nt)))
    else:
        pass

    print(study + " - done!")