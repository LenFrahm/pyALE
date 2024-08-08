import os
from os.path import isfile
import numpy as np
import nibabel as nb
from scipy import ndimage
from utils.kernel import kernel_conv
from utils.template import shape, pad_shape, prior, affine
from utils.compute import compute_ale


def contribution(exp_df, exp_name, tasks, tfce_enabled=True):
    
    cwd = os.getcwd()
    exp_idxs = list(range(exp_df.shape[0]))
    ma = np.stack(exp_df.MA.values)
    ale = compute_ale(ma)
    
    if tfce_enabled:
        corr_methods = ["TFCE", "FWE", "cFWE"]
    else:
        corr_methods = ["FWE", "cFWE"]
    
    for corr_method in corr_methods:
        txt = open(f"{cwd}/Results/MainEffect/Full/Contribution/{exp_name}_{corr_method}.txt", "w+")
        txt.write(f"\nStarting with {exp_name}! \n")
        txt.write(f"\n{exp_name}: {len(exp_idxs)} experiments; {exp_df.Subjects.sum()} unique subjects (average of {exp_df.Subjects.mean():.1f} per experiment) \n")
        
        if isfile(f"{cwd}/Results/MainEffect/Full/Volumes/Corrected/{exp_name}_{corr_method}05.nii"):
            # load in results that are corrected by the specific method
            results = nb.load(f"{cwd}/Results/MainEffect/Full/Volumes/Corrected/{exp_name}_{corr_method}05.nii").get_fdata()
            results = np.nan_to_num(results)
            if results.any() > 0:
                # cluster the significant voxels
                labels, cluster_count = ndimage.label(results)
                label, count = np.unique(labels, return_counts=True)
            
                for index, label in enumerate(np.argsort(count)[::-1][1:]):
                    cluster_idxs = np.vstack(np.where(labels == label))
                    cluster_size = cluster_idxs.shape[1]
                    # find the center voxel of the cluster
                    # to take the dot product with the affine matrix a row of 1s needs to be added the cluster indices (np.pad)
                    center = np.median(np.dot(affine, np.pad(cluster_idxs, ((0,1),(0,0)), constant_values=1)), axis=1)
                    if cluster_idxs[0].size > 5:
                        txt.write(f"\n\nCluster {index+1}: {cluster_size} voxel [Center: {int(center[0])}/{int(center[1])}/{int(center[2])}] \n")
            
                        ma_cluster_mask = ma[:,cluster_idxs[0],cluster_idxs[1], cluster_idxs[2]]
                        ale_cluster_mask = compute_ale(ma_cluster_mask)
                        contribution_arr = np.zeros((len(exp_idxs),4))
                        for idx in exp_idxs:
                            contribution_arr[idx,0] = np.sum(ma_cluster_mask[idx])
                            contribution_arr[idx,1] = 100*np.sum(ma_cluster_mask[idx])/cluster_size
                            proportion_of_ale = compute_ale(np.delete(ma_cluster_mask, idx, axis=0)) / ale_cluster_mask
                            contribution_arr[idx,2] = 100 * (1-np.mean(proportion_of_ale))
                            contribution_arr[idx,3] = 100 * (1-np.min(proportion_of_ale))
                            
                        contribution_arr[:,2] = contribution_arr[:,2]/np.sum(contribution_arr[:,2])*100
                        exp_idxs_sorted_by_contrib = np.argsort(contribution_arr[:,2])[::-1]
            
                        for idx in exp_idxs_sorted_by_contrib:
                            if contribution_arr[idx, 2]>.1 or contribution_arr[idx, 3]>5:
                                stx = list(" " * (exp_df.Author.str.len().max() + 2))
                                stx[0:len(exp_df.Author[idx])] = exp_df.Author[idx]
                                stx = "".join(stx)
                                n_subjects = exp_df.at[idx, "Subjects"]
                                txt.write(f"{stx}\t{contribution_arr[idx,0]:.3f}\t{contribution_arr[idx,1]:.3f}\t{contribution_arr[idx,2]:.2f}\t{contribution_arr[idx,3]:.2f}\t({n_subjects})\n")
            
                        txt.write("\n\n")
                        
                        # calculate task contribution to cluster
                        for i in range(tasks.shape[0]):
                            stx = list(" " * (tasks.Name.str.len().max()))
                            stx[0:len(tasks.Name[i])] = tasks.Name[i]
                            stx = "".join(stx)
                            mask = [s in tasks.ExpIndex[i] for s in exp_idxs]
                            if mask.count(True) >= 1:
                                contribution_task = np.sum(contribution_arr[mask], axis=0)
                                if contribution_task[0] > 0.01:
                                    txt.write(f"{stx}\t{contribution_task[0]:.3f}\t{contribution_task[1]:.3f}\t{contribution_task[2]:.2f}\t \n")
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