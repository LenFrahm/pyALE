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
from simulate_noise import simulate_noise
from scipy import ndimage
from joblib import Parallel, delayed
from kernel import kernel_conv


EPS =  np.finfo(float).eps
UC = 0.001
STEP = 10000


def plot_and_save(arr, prior, affine, img_folder, nii_folder):
    # Function that takes brain array and transforms it to NIFTI1 format
    # Saves it both as a statmap png and as a Nifti file
    arr[~prior] = np.nan
    nii_img = nb.Nifti1Image(arr, affine)
    plotting.plot_stat_map(nii_img, output_file=img_folder)
    nb.save(foci_img, nii_folder)

    
def illustrate_foci(exp_df, s0, template_shape, affine):    
    # Initialize empty array
    foci_arr np.zeros(template_shape)    
    # Load all peaks associated with study
    nested_list = [exp_df.XYZ[i].T[:,:3].tolist() for i in s0]
    flat_list = np.array([item for sublist in nested_list for item in sublist])    
    #Set all points in foci_arr that are peaks for the study to 1
    foci_arr[tuple(flat_list.T)] += 1
    
    return foci_arr


def compute_ale(exp_df, prior, bin_edge, template_shape, pad_template_shape):
    # initialize both an array for the ale score (brain template shape) and an empty histogram (studies * bins)
    ale = np.ones(template_shape)
    hx = np.zeros((len(s0),len(bin_edge)))
    # iterate through studies
    for i in s0:
        # create modelled activation map, smoothing each peak by the study specific gaussian kernel
        data = kernel_conv(peaks = exp_df.at[i, "XYZ"].T[:,:3],
                           kernel = exp_df.at[i, "Kernel"],
                           shape = pad_template_shape)
        # bin MA values and fill count into histogram
        bin_idxs, counts = np.unique(np.digitize(data[prior], bin_edge),return_counts=True)
        hx[i,bin_idxs] = counts
        # update ALE map
        ale = np.multiply(ale, 1-data)
    # flip ALE values; now > is better
    ale = 1-ale
    
    return ale, hx
 
    
def compute_cnull(hx,s0,bin_center):    
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
        ale_hist = np.zeros((len(bin_center),))
        #iterate over bins, which contain values
        for i in range(len(da2)):
            p = v2[da2[i]]*v1[da1]
            score = 1-(1-bin_center[da2[i]])*(1-bin_center[da1])
            ale_bin = np.round(score*step).astype(int)
            ale_hist[ale_bin] = np.add(ale_hist[ale_bin], p)
    last_used = np.where(ale_hist>0)[0][-1]
    c_null = np.flip(np.cumsum(np.flip(ale_hist[:last_used+1])))
    
    return ale_hist, last_used, c_null


def compute_z(ale, step, c_null):    
    # computing the corresponding histogram bin for each ale value
    ale_step = np.round(ale*step).astype(int)
    # replacing histogram bin number with corresponding histogram value (= p-value)
    p = np.array([c_null[i] for i in ale_step])
    # replace p-values that are smaller than the precision of floats with the precision of floats
    p[p < EPS] = EPS    
    # calculate z-values by plugging 1-p into a probability density function
    z = norm.ppf(1-p)
    
    return z


def compute_tfce(z):    
    # initialize empty array
    tfce = np.zeros(z.shape)
    # calculate step-size
    delta_t = np.max(z)/100
    # calculate tfce values using the parallelized function; opens 100 threads, 1 for each step
    vals, masks = zip(*Parallel(n_jobs=100, backend="threading")
                      (delayed(tfce_par(invol=z, h=h) for h in np.arange(0, np.max(z), delta_t)))
    # Parallelization makes it necessary to integrate the results afterwards
    # Each repition creats it's own mask and an amount of values corresponding to that mask
    for i in range(len(vals)):
        tfce[masks[i]] += vals[i]
    
    return tfce


def compute_cluster(z, cut_clust=None):    
    # disregard all voxels that feature a z-value of lower than 3.09 (approx. 3 standard deviations aways from the mean)
    # this serves as a preliminary thresholding
    z[z < norm.ppf(1-UC)] = 0
    sig_coords = np.where(z > 0)
    sig_arr = np.zeros(z.shape)
    sig_arr[tuple(sig_coords)] = 1
    # find clusters of significant z-values
    labels, cluster_count = ndimage.label(sig_arr)
    if cut_clust is not None:
        # check significance of cluster against the 95th percentile of the null distribution cluster size
        sig_clust = np.where(np.bincount(labels[labels > 0]) > cut_clust)[0]
        # z-value array that only features values for voxels that belong to significant clusters
        z = z*np.isin(labels, sig_clust)
        return z, max_clust
    # save number of voxels in biggest cluster
    max_clust = np.max(np.bincount(labels[labels>0]))
    
    return max_clust
    
    
def simulate_noise(sample_space, s0, num_peaks, kernels, c_null, delta_t, template_shape, pad_template_shape, tfce_sim=True)    
    # compute ALE values based on random peak locations sampled from a give sample_space
    # sample space could be all grey matter or only foci reported in brainmap
    vx = np.zeros((len(s0), sample_space.shape[1]))
    for i in s0:
        sample_peaks = sample_space[:,np.random.randint(0,sample_space.shape[1], num_peaks[i])]
        ale = kernel_conv(sample_peaks, kernels[i], pad_tmp_shape)
        vx[i,:] = data[sample_space[0], sample_space[1], sample_space[2]]
    ale = 1-np.prod(1-vx, axis=0)
    # Peak ALE threshold
    max_ale = np.max(ale)    
    # Cluster level threshold
    z = compute_z(ale, STEP, c_null)
    max_cluster = compute_cluster(z)    
    if tfce_sim == True:
        tfce = compute_tfce(z)
        # TFCE threshold
        max_tfce = np.max(tfce)
        
        return max_ale, max_cluster, max_tfce
                      
    else:
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
    
    # Load brain tempalte and declare a few frequently used variables belonging to the template
    template = nb.load(f"{mask_folder}Grey10.nii")
    template_data = template.get_fdata()
    template_shape = template_data.shape
    prior = np.zeros(template_shape, dtype=bool)
    prior[template_data > 0.1] = 1
    affine = template.affine
    pad_template_shape = [value+30 for value in template_shape]
    # Laoding in MNI template for later imaging purposes
    bg_img = nb.load(f"{mask_folder}MNI152.nii")

    
    # Declare variables for future calculations
    # simple list containing numbers 0 to number of studies -1 for iteration over studies
    s0 = list(range(exp_df.shape[0]))
    # highest possible ale value if every study had a peak at the same location.
    mb = 1
    for i in s0:
        mb = mb*(1-np.max(exp_df.at[i, 'Kernel']))
    
    # define bins for histogram
    bin_edge = np.arange(0.00005,1-mb+0.001,0.0001)
    bin_center = np.arange(0,1-mb+0.001,0.0001)

    """ Foci Illustration """
    
    # Check if Foci have already been illustrated
    if isfile(f'{cwd}/ALE/Foci/{exp_name}.nii'):
        print(f'{exp_name} - loading Foci')
        foci_arr = nb.load(f'{cwd}/ALE/Foci/{exp_name}.nii').get_fdata()        
    else:
        print(f'{study} - illustrate Foci')
        # take all peaks of included studies and indicate them with a one in a brain array
        foci_arr = illustrate_foci(exp_df, s0, template_shape, affine)
        # save both a .nii and a .png version
        plot_and_save(foci_arr, prior, affine, img_folder=f'{cwd}/ALE/Images/Foci/{exp_name}.png',
                                               nii_folder=f'{cwd}/ALE/Foci/{exp_name}.nii')
    
    """ ALE calculation """
    
    # Check if ALE scores have already been calculated
    if isfile(f'{cwd}/ALE/NullDistributions/{exp_name}.pickle'):
        print(f'{exp_name} - loading ALE')
        print(f'{exp_name} - loading null PDF')
        ale = nb.load(f'{cwd}/ALE/Volumes/{exp_name}.nii').get_fdata()
        ale = np.nan_to_num(ale)
        with open(f'{cwd}/ALE/NullDistributions/{exp_name}.pickle', 'rb') as f:
            ale_hist, last_used, c_null = pickle.load(f)            
    else:
        print(f'{study} - computing ALE')
        # Calculate ALE scores and create ale value histograms for each study
        ale, hx = compute_ale(exp_df, prior, bin_edge, template_shape, pad_template_shape)
        # save the ALE scores in both a .nii and a .png version
        plot_and_save(ale, prior, affine, img_folder=f"ALE/Images/ALE/{exp_name}.png",
                                          nii_folder=f'{cwd}/ALE/Volumes/{exp_name}.nii')
        
        print(f'{exp_name} - permutation-null PDF')
        # Use the histograms from above to estimate a null probability density function
        ale_hist, last_used, c_null = compute_cnull(hx,s0,bin_center)

        # Save the ale histogram and the null pdf to pickle
        pickle_object = (ale_hist, last_used, c_null)
        with open(f'{cwd}/ALE/NullDistributions/{exp_name}.pickle', "wb") as f:
            pickle.dump(pickle_object, f)
    
    """ TFCE calculation """
    
    # Check if TFCE scores have already been calculated
    if isfile(f'{cwd}/ALE/VolumesTFCE/{exp_name}.nii'):
        print(f'{exp_name} - loading p-values')

        z = nb.load(f'{cwd}/ALE/VolumesZ/{exp_name}.nii').get_fdata()
        z = np.nan_to_num(z)

        tfce_arr = nb.load(f'{cwd}/ALE/VolumesTFCE/{exp_name}.nii').get_fdata()
        tfce_arr = np.nan_to_num(tfce_arr)        
    else:
        print(f'{exp_name} - computing p-values')
        z = compute_z(ale, STEP, c_null)
        plot_and_save(z, prior, affine, img_folder=f"ALE/Images/Z/{exp_name}.png",
                                        nii_folder=f"ALE/VolumesZ/{exp_name}.nii") 
        tfce = compute_tfce(z)
        plot_and_save(tfce, prior, affine, img_folder=f"ALE/Images/TFCE/{exp_name}.png",
                                           nii_folder=f"ALE/VolumesTFCE/{exp_name}.nii")
        
    """ Null distribution calculation """
    
    # Check if Null distribution has already been calculated
    if isfile(f"{cwd}/ALE/NullDistributions/{exp_name}_clustP.pickle"):
        print(f'{exp_name} - loading noise')
        with open(f"{cwd}/ALE/NullDistributions/{exp_name}_clustP.pickle", 'rb') as f:
            nm, nn, nt = pickle.load(f)            
    else:
        print(f'{exp_name} - simulating noise')
        
        # Loading possible peak locations from matlab file
        permSpace5 = loadmat("MaskenEtc/permSpace5.mat")
        sample_space = permSpace5["allXYZ"]
        delta_t = np.max(z)/100
        
        # Simulate 19 experiments, which have the same amount of peaks as the original meta analysis but the
        # peaks are randomly distributed in the sample space. Then calculate all metrics that have
        # been calculated for the 'actual' data to create a null distribution unde the assumption of indipendence of results
        max_ale, max_cluster, max_tfce = zip(*Parallel(n_jobs=3, verbose=1)(delayed(simulate_noise)(sample_space = sample_space,
                                                                                s0 = s0,
                                                                                num_peaks = experiments.loc[:,'Peaks'],
                                                                                kernels = experiments.loc[:,'Kernel'],
                                                                                c_null = c_null,
                                                                                delta_t = delta_t) for i in range(noise_repeat)))
        # save simulation results to pickle
        simulation_pickle = (max_ale, max_cluster, max_tfce)
        with open(f"{cwd}/ALE/NullDistributions/{exp_name}_clustP.pickle", "wb") as f:
            pickle.dump(simulation_pickle, f)
            
    """ Multiple comparison error correction: FWE, cFWE, TFCE """

    if not isfile(f"{cwd}/ALE/Results/{exp_name}_TFCE05.nii'):
        print(f'{exp_name} - inference and printing')
        # voxel wise family wise error correction
        cut_max = np.percentile(nm, 95)
        ale = ale*(ale > cut_max)
        plot_and_save(ale, prior, affine, img_folder=f"ALE/Images/{exp_name}_FWE05.png",
                                          nii_folder=f"ALE/Results/{exp_name}_FWE05.nii")
        print(f"Min p-value for FWE:{sum(nm>np.max(ale))/len(nn)}")
                  
        # cluster wise family wise error correction
        cut_clust = np.percentile(nn, 95)                  
        z, max_clust = compute_cluster(z, cut_clust)
        plot_and_save(z, prior, affine, img_folder=f"ALE/Images/{exp_name}_cFWE05.png",
                                        nii_folder=f"ALE/Results/{exp_name}_cFWE05.nii")
        print(f"Min p-value for cFWE:{sum(nn>max_clust)/len(nm)}")

        # tfce error correction
        cut_tfce = np.percentile(nt, 95)
        tfce = tfce*(tfce > cut_tfce)
        plot_and_save(tfce, prior, affine, img_folder=f"ALE/Images/{exp_name}_TFCE05.png",
                                           nii_folder=f"ALE/Results/{exp_name}_TFCE05.nii")
        print(f"Min p-value for TFCE:{sum(nt>np.max(tfce_arr))/len(nt)}")
                  
    else:
        pass

    print(f"{exp_name} - done!")