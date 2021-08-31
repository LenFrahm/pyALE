import os
from os.path import isfile
import numpy as np
import nibabel as nb
from scipy import ndimage
from utils.kernel import kernel_conv
from utils.template import shape, pad_shape, prior, affine


def contribution(exp_df, exp_name, exp_idxs, tasks, tfce_enabled=True):
    
    cwd = os.getcwd()
    s0 = list(range(exp_df.shape[0]))
    ma = np.stack(exp_df.MA.values)
    
    if tfce_enabled:
        corr_methods = ["TFCE", "FWE", "cFWE"]
    else:
        corr_methods = ["FWE", "cFWE"]
    
    for corr_method in corr_methods:
        txt = open(f"{cwd}/Results/MainEffect/Full/Contribution/{exp_name}_{corr_method}.txt", "w+")
        txt.write(f"\nStarting with {exp_name}! \n")
        txt.write(f"\n{exp_name}: {len(s0)} experiments; {exp_df.Subjects.sum()} unique subjects (average of {exp_df.Subjects.mean():.1f} per experiment) \n")
        
        if isfile(f"{cwd}/Results/MainEffect/Full/Volumes/Corrected/{exp_name}_{corr_method}05.nii"):
            # load in results that are corrected by the specific method
            results = nb.load(f"{cwd}/Results/MainEffect/Full/Volumes/Corrected/{exp_name}_{corr_method}05.nii").get_fdata()
            results = np.nan_to_num(results)
            if results.any() > 0:    
                ale = nb.load(f"{cwd}/Results/MainEffect/Full/Volumes/ALE/{exp_name}.nii")
                # cluster the significant voxels
                labels, cluster_count = ndimage.label(results)
                label, count = np.unique(labels, return_counts=True)

                for index, label in enumerate(np.argsort(count)[::-1][1:]):
                    clust_ind = np.vstack(np.where(labels == label))
                    clust_size = clust_ind.shape[1]
                    # find the center voxel of the cluster
                    # to take the dot product with the affine matrix a row of 1s needs to be added the cluster indices (np.pad)
                    center = np.median(np.dot(affine, np.pad(clust_ind, ((0,1),(0,0)), constant_values=1)), axis=1)
                    if clust_ind[0].size > 5:
                        txt.write(f"\n\nCluster {index+1}: {clust_size} voxel [Center: {int(center[0])}/{int(center[1])}/{int(center[2])}] \n")
                        
                        # calculate the relative contribution of each study to the cluster voxels total ALE
                        ax = ma[:, clust_ind[0], clust_ind[1], clust_ind[2]]
                        axf = 1-np.prod(1-ax, axis=0) # ale values
                        axr = np.array([1-np.prod(1-np.delete(ax, i, axis=0), axis=0) for i in s0]) # ale values if one study would be omitted
                        wig = np.array([np.sum(ma[i][tuple(clust_ind)]) for i in s0]) # summing MA over the cluster per study
                        # summarizing array for each study: 1. total MA in cluster 2. average MA per voxel in cluster
                        # 3. relative contribution to cluster ale 4. maximum ale contribution in the cluster (single voxel)
                        xsum = np.array([[wig[i], 100*wig[i]/clust_size, 100*(1-np.mean(np.divide(axr[i,:], axf))), np.max(100*(1-np.divide(axr[i,:], axf)))] for i in s0])
                        # convert relative contribtuion to percentages that add to 100
                        xsum[:,2] = xsum[:,2]/np.sum(xsum[:,2])*100

                        for i in s0:
                            if xsum[i, 2]>.1 or xsum[i, 3]>5:
                                stx = list(" " * (exp_df.Author.str.len().max() + 2))
                                stx[0:len(exp_df.Author[i])] = exp_df.Author[i]
                                stx = "".join(stx)
                                n_subjects = exp_df.at[i, "Subjects"]
                                txt.write(f"{stx}\t{xsum[i,0]:.3f}\t{xsum[i,1]:.3f}\t{xsum[i,2]:.2f}\t{xsum[i,3]:.2f}\t({n_subjects})\n")

                        txt.write("\n\n")
                        
                        # calculate task contribution to cluster
                        for i in range(tasks.shape[0]):
                            stx = list(" " * (tasks.Name.str.len().max()))
                            stx[0:len(tasks.Name[i])] = tasks.Name[i]
                            stx = "".join(stx)
                            mask = [s in tasks.ExpIndex[i] for s in exp_idxs]
                            if mask.count(True) > 1:
                                xsum_tmp = np.sum(xsum[mask], axis=0)
                                txt.write(f"{stx}\t{xsum_tmp[0]:.3f}\t{xsum_tmp[1]:.3f}\t{xsum_tmp[2]:.2f}\t \n")
                            elif mask.count(True) == 1:
                                txt.write(f"{stx}\t{xsum_tmp[0]:.3f}\t{xsum_tmp[1]:.3f}\t{xsum_tmp[2]:.2f}\t \n")
                            else:
                                pass

                txt.write(f"\nDone with {corr_method}!")
                txt.close()
            else:
                txt.write(f"No significant clusters in {corr_method}!")
                txt.close()
        else:
            txt.write(f"Could not find {corr_method} results for {exp_name}!")
            txt.close()