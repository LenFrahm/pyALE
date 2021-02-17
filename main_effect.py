import os
from os.path import isfile
import pandas as pd
import numpy as np
from scipy.stats import norm
from nilearn import plotting
import nibabel as nb
import pickle
from compile_studies import compile_studies
from tfce_par import tfce_par
from scipy.io import loadmat
from scipy import ndimage
from joblib import Parallel, delayed
from kernel import kernel_conv
# importing brain template information
from template import shape, pad_shape, prior, affine


EPS =  np.finfo(float).eps
THRESH = 0.001
STEP = 10000
REPEAT = 3000


def plot_and_save(arr, img_folder, nii_folder):
    # Function that takes brain array and transforms it to NIFTI1 format
    # Saves it both as a statmap png and as a Nifti file
    arr[~prior] = np.nan
    nii_img = nb.Nifti1Image(arr, affine)
    plotting.plot_stat_map(nii_img, output_file=img_folder)
    nb.save(nii_img, nii_folder)
    arr = np.nan_to_num(arr)
    
    return arr

    
def illustrate_foci(s0, peaks):    
    foci_arr = np.zeros(shape)    
    # Load all peaks associated with study
    peaks = np.concatenate(peaks)
    #Set all points in foci_arr that are peaks for the study to 1
    foci_arr[tuple(peaks.T)] += 1
    
    return foci_arr


def compute_ale(s0, peaks, kernels, bin_edges):
    ale = np.ones(shape)
    hx = np.zeros((len(s0),len(bin_edges)))
    # iterate through studies
    for i in s0:
        # create modelled activation map, smoothing each peak by the study specific gaussian kernel
        data = kernel_conv(peaks = peaks[i],
                           kernel = kernels[i])
        # bin MA values and fill count into histogram
        bin_idxs, counts = np.unique(np.digitize(data[prior], bin_edges),return_counts=True)
        hx[i,bin_idxs] = counts
        # update ALE map
        ale = np.multiply(ale, 1-data)
    # flip ALE values; now > is better
    ale = 1-ale
    
    return ale, hx
 
    
def compute_cnull(s0, hx, bin_centers):    
    ale_hist = hx[0,:]
    for i in range(1,len(s0)):
        v1 = ale_hist
        # save bins, which there are entries in the combined hist
        da1 = np.where(v1 > 0)[0]
        # normalize combined hist to sum to 1
        v1 = ale_hist/np.sum(v1)
        
        v2 = hx[i,:]
        # save bins, which there are entries in the study hist
        da2 = np.where(v2 > 0)[0]
        # normalize study hist to sum to 1
        v2 = hx[i,:]/np.sum(v2)
        ale_hist = np.zeros((len(bin_centers),))
        #iterate over bins, which contain values
        for i in range(len(da2)):
            p = v2[da2[i]]*v1[da1]
            score = 1-(1-bin_centers[da2[i]])*(1-bin_centers[da1])
            ale_bin = np.round(score*STEP).astype(int)
            ale_hist[ale_bin] = np.add(ale_hist[ale_bin], p)
    last_used = np.where(ale_hist>0)[0][-1]
    c_null = np.flip(np.cumsum(np.flip(ale_hist[:last_used+1])))
    
    return ale_hist, last_used, c_null


def compute_z(ale, c_null):    
    # computing the corresponding histogram bin for each ale value
    ale_step = np.round(ale*STEP).astype(int)
    # replacing histogram bin number with corresponding histogram value (= p-value)
    p = np.array([c_null[i] for i in ale_step])
    p[p < EPS] = EPS    
    # calculate z-values by plugging 1-p into a probability density function
    z = norm.ppf(1-p)
    
    return z


def compute_tfce(z, sample_space, delta_t):
    
    if z.shape != shape:
        tmp = np.zeros(shape)
        tmp[tuple(sample_space)] = z
        z = tmp
    
    lc = np.min(sample_space, axis=1)
    uc = np.max(sample_space, axis=1)  
    z = z[lc[0]:uc[0],lc[1]:uc[1],lc[2]:uc[2]]
    
    tmp = np.zeros(z.shape)
    # calculate tfce values using the parallelized function
    vals, masks = zip(*Parallel(n_jobs=4, backend="threading")
                     (delayed(tfce_par)(invol=z, h=h) for h in np.arange(0, np.max(z), delta_t)))
    # Parallelization makes it necessary to integrate the results afterwards
    # Each repition creats it's own mask and an amount of values corresponding to that mask
    for i in range(len(vals)):
        tmp[masks[i]] += vals[i]
        
    tfce = np.zeros(shape)
    tfce[lc[0]:uc[0],lc[1]:uc[1],lc[2]:uc[2]] = tmp
    
    return tfce


def compute_cluster(z, thresh=THRESH, sample_space=None, cut_clust=None):    
    # disregard all voxels that feature a z-value of lower than 3.09 (approx. 3 standard deviations aways from the mean)
    # this serves as a preliminary thresholding
    z_copy = z.copy()
    z_copy[z_copy < norm.ppf(1-thresh)] = 0
    sig_coords = np.where(z_copy > 0)
    if sample_space is not None:
        sig_coords = sample_space[:,sig_coords]
    sig_arr = np.zeros(shape)
    sig_arr[tuple(sig_coords)] = 1
    # find clusters of significant z-values
    labels, cluster_count = ndimage.label(sig_arr)
    # save number of voxels in biggest cluster
    max_clust = np.max(np.bincount(labels[labels>0]))
    if cut_clust is not None:
        # check significance of cluster against the 95th percentile of the null distribution cluster size
        sig_clust = np.where(np.bincount(labels[labels > 0]) > cut_clust)[0]
        # z-value array that only features values for voxels that belong to significant clusters
        z_copy = z_copy*np.isin(labels, sig_clust)
        return z_copy, max_clust
    
    return max_clust
    
    
def compute_null(s0, sample_space, num_peaks, kernels, bin_centers, bin_edges, c_null, delta_t=None, tfce=None, thresh=THRESH):
    # compute ALE values based on random peak locations sampled from a give sample_space
    # sample space could be all grey matter or only foci reported in brainmap
    noise_peaks = np.array([sample_space[:,np.random.randint(0,sample_space.shape[1], num_peaks[i])].T for i in s0], dtype=object)
    ale, hx = compute_ale(s0, noise_peaks, kernels, bin_edges)
    if c_null == "simulate":
        ale_hist, last_used, c_null = compute_cnull(s0, hx, bin_centers)
        thresh = 1-bin_centers[np.where(c_null<0.001)[0][0]]
    # Peak ALE threshold
    max_ale = np.max(ale)    
    z = compute_z(ale, c_null)
    if delta_t == None:
        delta_t = np.max(z)/100
    # Cluster level threshold
    max_cluster = compute_cluster(z, thresh, sample_space)
    if tfce == "simulate":
        tfce = compute_tfce(z, sample_space, delta_t)
        # TFCE threshold
        max_tfce = np.max(tfce)
        return max_ale, max_cluster, max_tfce
        
    return max_ale, max_cluster

   
    
def main_effect(exp_df, exp_name):    
    # Create necessary folder structure
    cwd = os.getcwd()
    mask_folder = f"{cwd}/MaskenEtc/"
    folders_req = ['Volumes', 'NullDistributions', 'VolumesZ', 'VolumesTFCE', 'Results', 'Images', 'Foci']
    folders_req_imgs = ['Foci', 'ALE', 'TFCE', 'Z']
    try:
        os.mkdir('ALE')
        for folder in folders_req:
            os.mkdir(f'ALE/{folder}')
            if folder == 'Images':
                for sub_folder in folders_req_imgs:
                    os.mkdir(f'ALE/Images/{sub_folder}')
    except FileExistsError:
        pass

    # Declare variables for future calculations
    # simple list containing numbers 0 to number of studies -1 for iteration over studies
    s0 = list(range(exp_df.shape[0]))
    # highest possible ale value if every study had a peak at the same location.
    mb = 1
    for i in s0:
        mb = mb*(1-np.max(exp_df.at[i, 'Kernel']))
    
    # define bins for histogram
    bin_edges = np.arange(0.00005,1-mb+0.001,0.0001)
    bin_centers = np.arange(0,1-mb+0.001,0.0001)
    
    sample_space = np.array(np.where(prior == 1))
    
    peaks = np.array([exp_df.XYZ[i].T[:,:3] for i in s0], dtype=object)
    

    """ Foci Illustration """
    
    if isfile(f'{cwd}/ALE/Foci/{exp_name}.nii'):
        print(f'{exp_name} - loading Foci')
        foci_arr = nb.load(f'{cwd}/ALE/Foci/{exp_name}.nii').get_fdata()        
    else:
        print(f'{exp_name} - illustrate Foci')
        # take all peaks of included studies and indicate them with a one in a brain array
        foci_arr = illustrate_foci(s0, peaks)
        # save both a .nii and a .png version
        foci_arr = plot_and_save(foci_arr, img_folder=f'{cwd}/ALE/Images/Foci/{exp_name}.png', 
                                           nii_folder=f'{cwd}/ALE/Foci/{exp_name}.nii')
    
    """ ALE calculation """
    
    if isfile(f'{cwd}/ALE/NullDistributions/{exp_name}.pickle'):
        print(f'{exp_name} - loading ALE')
        print(f'{exp_name} - loading null PDF')
        ale = nb.load(f'{cwd}/ALE/Volumes/{exp_name}.nii').get_fdata()
        ale = np.nan_to_num(ale)
        with open(f'{cwd}/ALE/NullDistributions/{exp_name}.pickle', 'rb') as f:
            ale_hist, last_used, c_null = pickle.load(f)            
    else:
        print(f'{exp_name} - computing ALE')
        # Calculate ALE scores and create ale value histograms for each study
        ale, hx = compute_ale(s0, peaks, exp_df.Kernel, bin_edges)
        # save the ALE scores in both a .nii and a .png version
        ale = plot_and_save(ale, img_folder=f"ALE/Images/ALE/{exp_name}.png",
                                 nii_folder=f'{cwd}/ALE/Volumes/{exp_name}.nii')
        
        print(f'{exp_name} - permutation-null PDF')
        # Use the histograms from above to estimate a null probability density function
        ale_hist, last_used, c_null = compute_cnull(s0, hx, bin_centers)

        # Save the ale histogram and the null pdf to pickle
        pickle_object = (ale_hist, last_used, c_null)
        with open(f'{cwd}/ALE/NullDistributions/{exp_name}.pickle', "wb") as f:
            pickle.dump(pickle_object, f)
    
    """ TFCE calculation """
    
    if isfile(f'{cwd}/ALE/VolumesTFCE/{exp_name}.nii'):
        print(f'{exp_name} - loading p-values & TFCE')

        z = nb.load(f'{cwd}/ALE/VolumesZ/{exp_name}.nii').get_fdata()
        z = np.nan_to_num(z)
        delta_t = np.max(z)/100

        tfce = nb.load(f'{cwd}/ALE/VolumesTFCE/{exp_name}.nii').get_fdata()
        tfce= np.nan_to_num(tfce)        
    else:
        print(f'{exp_name} - computing p-values & TFCE')
        z = compute_z(ale, c_null)
        delta_t = np.max(z)/100
        tfce = compute_tfce(z, sample_space, delta_t)
        
        z = plot_and_save(z, img_folder=f"ALE/Images/Z/{exp_name}.png",
                             nii_folder=f"ALE/VolumesZ/{exp_name}.nii") 
        tfce = plot_and_save(tfce, img_folder=f"ALE/Images/TFCE/{exp_name}.png",
                                   nii_folder=f"ALE/VolumesTFCE/{exp_name}.nii")
        
    """ Null distribution calculation """
    
    if isfile(f"{cwd}/ALE/NullDistributions/{exp_name}_clustP.pickle"):
        print(f'{exp_name} - loading noise')
        with open(f"{cwd}/ALE/NullDistributions/{exp_name}_clustP.pickle", 'rb') as f:
            max_ale, max_cluster, max_tfce = pickle.load(f)            
    else:
        print(f'{exp_name} - simulating noise')       
        # Simulate 19 experiments, which have the same amount of peaks as the original meta analysis but the
        # peaks are randomly distributed in the sample space. Then calculate all metrics that have
        # been calculated for the 'actual' data to create a null distribution unde the assumption of indipendence of results
        max_ale, max_cluster, max_tfce = zip(*Parallel(n_jobs=3, verbose=1)(delayed(compute_null)(s0 = s0,
                                                                                                  sample_space = sample_space,
                                                                                                  num_peaks = exp_df.Peaks,
                                                                                                  kernels = exp_df.Kernel,
                                                                                                  bin_centers = bin_centers,
                                                                                                  bin_edges = bin_edges,
                                                                                                  c_null = c_null,
                                                                                                  delta_t = delta_t,
                                                                                                  tfce = "simulate") for i in range(REPEAT)))
        # save simulation results to pickle
        simulation_pickle = (max_ale, max_cluster, max_tfce)
        with open(f"{cwd}/ALE/NullDistributions/{exp_name}_clustP.pickle", "wb") as f:
            pickle.dump(simulation_pickle, f)
            
    """ Multiple comparison error correction: FWE, cFWE, TFCE """

    if not isfile(f"{cwd}/ALE/Results/{exp_name}_TFCE05.nii"):
        print(f'{exp_name} - inference and printing')
        # voxel wise family wise error correction
        cut_max = np.percentile(max_ale, 95)
        ale = ale*(ale>cut_max)
        ale = plot_and_save(ale, img_folder=f"ALE/Images/{exp_name}_FWE05.png",
                                 nii_folder=f"ALE/Results/{exp_name}_FWE05.nii")
        print(f"Min p-value for FWE:{sum(max_ale>np.max(ale))/len(max_ale)}")
                  
        # cluster wise family wise error correction
        cut_clust = np.percentile(max_cluster, 95)                  
        z, max_clust = compute_cluster(z, cut_clust=cut_clust)
        z = plot_and_save(z, img_folder=f"ALE/Images/{exp_name}_cFWE05.png",
                             nii_folder=f"ALE/Results/{exp_name}_cFWE05.nii")
        print(f"Min p-value for cFWE:{sum(max_cluster>max_clust)/len(max_cluster)}")

        # tfce error correction
        cut_tfce = np.percentile(max_tfce, 95)
        tfce = tfce*(tfce>cut_tfce)
        tfce = plot_and_save(tfce, img_folder=f"ALE/Images/{exp_name}_TFCE05.png",
                                   nii_folder=f"ALE/Results/{exp_name}_TFCE05.nii")
        print(f"Min p-value for TFCE:{sum(max_tfce>np.max(tfce))/len(max_tfce)}")
                  
    else:
        pass

    print(f"{exp_name} - done!")