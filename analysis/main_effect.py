import os
from os.path import isfile
import pandas as pd
import numpy as np
import nibabel as nb
import pickle
from joblib import Parallel, delayed
from utils.template import sample_space
from utils.compute import *
     
def main_effect(exp_df, exp_name, bin_steps=0.0001, cluster_thresh=0.001, null_repeats=5000, target_n=None, sample_n=None):
    # Declare variables for future calculations
    # simple list containing numbers 0 to number of studies -1 for iteration over studies
    s0 = list(range(exp_df.shape[0]))
    # highest possible ale value if every study had a peak at the same location.
    mb = 1
    for i in s0:
        mb = mb*(1-np.max(exp_df.at[i, 'Kernels']))
    
    # define bins for histogram
    bin_edges = np.arange(0.00005,1-mb+0.001,bin_steps)
    bin_centers = np.arange(0,1-mb+0.001,bin_steps)
    step = int(1/bin_steps)
    
    peaks = np.array([exp_df.XYZ[i].T for i in s0], dtype=object)
    

    """ Foci Illustration """
    
    if not isfile(f'Results/MainEffect/Full/Volumes/Foci/{exp_name}.nii'):
        print(f'{exp_name} - illustrate Foci')
        # take all peaks of included studies and indicate them with a one in a brain array
        foci_arr = illustrate_foci(peaks)
        foci_arr = plot_and_save(foci_arr, img_folder=f'Results/MainEffect/Full/Images/Foci/{exp_name}.png', 
                                           nii_folder=f'Results/MainEffect/Full/Volumes/Foci/{exp_name}.nii')
    
    
    ma = np.array([exp_df.MA.values[i] for i in s0])
    hx = compute_hx(s0, ma, bin_edges)
    
    if target_n:
        """ Probabilistic ALE """
        
        print(f"{exp_name} - entering probabilistic ALE routine.")
        
        if isfile(f"Results/MainEffect/CV/NullDistributions/{exp_name}_ccut_{target_n}.pickle"):
            print(f"{exp_name} - computing cv cluster cut-off.")
            with open(f"Results/MainEffect/CV/NullDistributions/{exp_name}_ccut_{target_n}.pickle", 'rb') as f:
                cut_cluster = pickle.load(f)
        else:            
            print(f"{exp_name} - computing cv cluster cut-off.")
            max_ale, max_cluster = zip(*Parallel(n_jobs=-1, verbose=1)(delayed(compute_null_cutoffs)(s0 = s0,
                                                                                                    sample_space = sample_space,
                                                                                                    num_peaks = exp_df.Peaks,
                                                                                                    kernels = exp_df.Kernels,
                                                                                                    step=step,
                                                                                                    thresh = cluster_thresh,
                                                                                                    bin_centers=bin_centers,
                                                                                                    bin_edges=bin_edges,
                                                                                                    target_n=target_n) for i in range(null_repeats)))
            cut_cluster = np.percentile(max_cluster, 95)
            with open(f"Results/MainEffect/CV/NullDistributions/{exp_name}_ccut_{target_n}.pickle", "wb") as f:
                pickle.dump(cut_cluster, f)
        
        if not isfile(f"Results/MainEffect/CV/Volumes/ALE/{exp_name}_{target_n}.nii"):     
            print(f"{exp_name} - computing cv ale.")
            samples = create_samples(s0, sample_n, target_n)
            sub_ale = Parallel(n_jobs=8, verbose=2)(delayed(compute_sub_ale)(samples[i], ma, hx, bin_centers, cut_cluster, thresh=cluster_thresh) for i in range(samples.shape[0]))
            sub_ale = np.mean(sub_ale, axis=0)
            sub_ale = plot_and_save(sub_ale, img_folder=f"Results/MainEffect/CV/Images/{exp_name}_{target_n}.png",
                                             nii_folder=f"Results/MainEffect/CV/Volumes/{exp_name}_{target_n}.nii")
        
        print(f"{exp_name} - probabilistic ALE done!")
        return
    
    else:
        
        """ Full ALE """
        
        """ ALE calculation """

        if isfile(f'Results/MainEffect/Full/NullDistributions/{exp_name}.pickle'):
            print(f'{exp_name} - loading ALE')
            print(f'{exp_name} - loading null PDF')
            ale = nb.load(f'Results/MainEffect/Full/Volumes/ALE/{exp_name}.nii').get_fdata()
            with open(f'Results/MainEffect/Full/NullDistributions/{exp_name}.pickle', 'rb') as f:
                hx_conv, _ = pickle.load(f)    
                
        else:
            print(f'{exp_name} - computing ALE and null PDF')
            ale = compute_ale(ma)
            ale = plot_and_save(ale, img_folder=f'Results/MainEffect/Full/Images/ALE/{exp_name}.png',
                                     nii_folder=f'Results/MainEffect/Full/Volumes/ALE/{exp_name}.nii')
            
            # Use the histograms from above to estimate a null probability density function
            hx_conv = compute_hx_conv(s0, hx, bin_centers, step)

            pickle_object = (hx_conv, hx)
            with open(f'Results/MainEffect/Full/NullDistributions/{exp_name}.pickle', "wb") as f:
                pickle.dump(pickle_object, f)

        """ TFCE calculation """

        if isfile(f'Results/MainEffect/Full/Volumes/TFCE/{exp_name}.nii'):
            print(f'{exp_name} - loading p-values & TFCE')
            z = nb.load(f'Results/MainEffect/Full/Volumes/Z/{exp_name}.nii').get_fdata()
            tfce = nb.load(f'Results/MainEffect/Full/Volumes/TFCE/{exp_name}.nii').get_fdata()   
            
        else:
            print(f'{exp_name} - computing p-values & TFCE')
            z = compute_z(ale, hx_conv, step)
            tfce = compute_tfce(z, sample_space)

            z = plot_and_save(z, nii_folder=f'Results/MainEffect/Full/Volumes/Z/{exp_name}.nii') 
            tfce = plot_and_save(tfce, img_folder=f'Results/MainEffect/Full/Images/TFCE/{exp_name}.png',
                                       nii_folder=f'Results/MainEffect/Full/Volumes/TFCE/{exp_name}.nii')

        """ Null distribution calculation """

        if isfile(f"Results/MainEffect/Full/NullDistributions/{exp_name}_null.pickle"):
            print(f'{exp_name} - loading null')
            with open(f"Results/MainEffect/Full/NullDistributions/{exp_name}_null.pickle", 'rb') as f:
                _, max_ale, max_cluster, max_tfce = pickle.load(f)            
        else:
            print(f'{exp_name} - simulating null')       
            # Simulate 19 experiments, which have the same amount of peaks as the original meta analysis but the
            # peaks are randomly distributed in the sample space. Then calculate all metrics that have
            # been calculated for the 'actual' data to create a null distribution unde the assumption of indipendence of results
            null_ale, max_ale, max_cluster, max_tfce = zip(*Parallel(n_jobs=-1, verbose=1)(delayed(compute_null_cutoffs)(s0 = s0,
                                                                                                                         sample_space = sample_space,
                                                                                                                         num_peaks = exp_df.Peaks,
                                                                                                                         kernels = exp_df.Kernels,
                                                                                                                         hx_conv = hx_conv,
                                                                                                                         tfce=1) for i in range(null_repeats)))
                    # save simulation results to pickle
            simulation_pickle = (null_ale, max_ale, max_cluster, max_tfce)
            with open(f"Results/MainEffect/Full/NullDistributions/{exp_name}_null.pickle", "wb") as f:
                pickle.dump(simulation_pickle, f)

        """ Multiple comparison error correction: FWE, cFWE, TFCE """

        if not isfile(f"Results/MainEffect/Full/Volumes/Corrected/{exp_name}_TFCE05.nii"):
            print(f'{exp_name} - inference and printing')
            # voxel wise family wise error correction
            cut_max = np.percentile(max_ale, 95)
            ale = ale*(ale>cut_max)
            ale = plot_and_save(ale, img_folder=f"Results/MainEffect/Full/Images/Corrected/{exp_name}_FWE05.png",
                                     nii_folder=f"Results/MainEffect/Full/Volumes/Corrected/{exp_name}_FWE05.nii")
            print(f"Min p-value for FWE:{sum(max_ale>np.max(ale))/len(max_ale)}")

            # cluster wise family wise error correction
            cut_cluster = np.percentile(max_cluster, 95)                  
            z, max_clust = compute_cluster(z, thresh=cluster_thresh, cut_cluster=cut_cluster)
            z = plot_and_save(z, img_folder=f"Results/MainEffect/Full/Images/Corrected/{exp_name}_cFWE05.png",
                                 nii_folder=f"Results/MainEffect/Full/Volumes/Corrected/{exp_name}_cFWE05.nii")
            print(f"Min p-value for cFWE:{sum(max_cluster>max_clust)/len(max_cluster)}")

            # tfce error correction
            cut_tfce = np.percentile(max_tfce, 95)
            tfce = tfce*(tfce>cut_tfce)
            tfce = plot_and_save(tfce, img_folder=f"Results/MainEffect/Full/Images/Corrected/{exp_name}_TFCE05.png",
                                       nii_folder=f"Results/MainEffect/Full/Volumes/Corrected/{exp_name}_TFCE05.nii")
            print(f"Min p-value for TFCE:{sum(max_tfce>np.max(tfce))/len(max_tfce)}")

        else:
            pass

        print(f"{exp_name} - done!")