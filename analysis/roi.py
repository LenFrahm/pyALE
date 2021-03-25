import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.compute import compute_ale, compute_perm_diff
from joblib import Parallel, delayed

def roi_ale(exp_df, exp_name, masks, mask_names, null_repeats, null_ale):
    s0 = list(range(exp_df.shape[0]))
    ma = np.stack(exp_df.MA.values)
    ale = compute_ale(ma)

    for index, mask in enumerate(masks):
        if np.unique(mask).shape[0] == 2:
            bench_sum = np.sum(ale[mask])
            bench_max = np.max(ale[mask])

            null_sum = np.sum(null_ale[:,mask], axis=1)
            null_max = np.max(null_ale[:,mask], axis=1)

            plot_roi_ale(exp_name, mask_names[index], bench_sum, bench_max, null_sum, null_max, null_repeats)
        else:
            for value in np.unique(mask)[1:]:
                bench_sum = np.sum(ale[mask == value])
                if bench_sum == 0:
                    continue
                bench_max = np.max(ale[mask == value])

                null_sum = np.sum(null_ale[:,mask == value], axis=1)
                null_max = np.max(null_ale[:,mask == value], axis=1)

                plot_roi_ale(exp_name, mask_names[index], bench_sum, bench_max, null_sum, null_max, null_repeats, index=value)

def roi_ale_contrast(exp_dfs, exp_names, masks, mask_names, null_repeats):
    s = [list(range(exp_dfs[i].shape[0])) for i in (0,1)]
    ma1 = np.stack(exp_dfs[0].MA.values)
    ale1 = compute_ale(ma1)

    ma2 = np.stack(exp_dfs[1].MA.values)
    ale2 = compute_ale(ma2)

    ma = 1 - np.vstack((ma1,ma2))
    perm_diff = Parallel(n_jobs=8, verbose=1)(delayed(compute_perm_diff)(s, ma) for i in range(null_repeats))
    
    for index, mask in enumerate(masks):
        if np.unique(mask).shape[0] == 2:
            bench_diff = ale1[mask] - ale2[mask]
            bench_sum = np.sum(bench_diff)
            bench_max = np.max(bench_diff)
            
            null_sum = [np.sum(perm_diff[i][mask]) for i in range(len(perm_diff))]
            null_max = [np.max(perm_diff[i][mask]) for i in range(len(perm_diff))]

            plot_roi_ale(exp_names, mask_names[index], bench_sum, bench_max, null_sum, null_max, null_repeats)
        
        else:
            for value in np.unique(mask)[1:]:
                bench_diff = ale1[mask==value] - ale2[mask==value]
                bench_sum = np.sum(bench_diff)
                if bench_sum == 0:
                    continue
                bench_max = np.max(bench_diff)
                
                null_sum = [np.sum(perm_diff[i][mask==value]) for i in range(len(perm_diff))]
                null_max = [np.max(perm_diff[i][mask==value]) for i in range(len(perm_diff))]
        
                plot_roi_ale(exp_names, mask_names[index], bench_sum, bench_max, null_sum, null_max, null_repeats, index=value)
            
            
def check_rois(exp_dfs, exp_names, masks, mask_names, null_repeats, null_ale=None):
    if len(exp_dfs) == 2:
        roi_ale_contrast(exp_dfs, exp_names, masks, mask_names, null_repeats)
    else:
        roi_ale(exp_dfs, exp_names, masks, mask_names, null_repeats, null_ale)
        
def plot_roi_ale(exp_names, mask_name, bench_sum, bench_max, null_sum, null_max, null_repeats, index="Full"):
    fig, ax = plt.subplots(1, 2, figsize=(15,10))
    fig.patch.set_facecolor('skyblue')

    weights = np.ones_like(null_sum)/float(len(null_sum))
    n, _, patches = ax[0].hist(null_sum,np.arange(0,np.ceil(np.max(null_sum)),np.max(null_sum)/50), weights=weights)

    ax[0].vlines(bench_sum,0,np.max(n)+np.max(n)/8,colors="r")
    p_value = (null_sum > bench_sum).sum() / null_repeats
    ax[0].annotate(f'Observed ALE integral \n p-value = {p_value}', xy=(bench_sum, np.max(n)+np.max(n)/8), ha='center')

    p05 = np.percentile(null_sum, 95)
    line1 = ax[0].vlines(p05, 0, np.max(n),colors="darkgreen", label="p < 0.05")
    p01 = np.percentile(null_sum, 99)
    line2 = ax[0].vlines(p01, 0, np.max(n), colors="green", label="p < 0.01")
    p001 = np.percentile(null_sum, 99.9)
    line3 = ax[0].vlines(p001, 0, np.max(n), colors="lime", label="p < 0.001")

    ax[0].set(xlabel="ALE integral in roi mask",ylabel=f"Percentage of {null_repeats} realizations")
    ax[0].title.set_text(f"{mask_name}_{index} - ALE integral")
    ax[0].legend(handles=[line1, line2, line3], loc='upper right')



    weights = np.ones_like(null_max)/float(len(null_max))
    n, _, patches = ax[1].hist(null_max,np.arange(0,np.max(null_max)+np.max(null_max)/6,np.max(null_max)/50), weights=weights)

    ax[1].vlines(bench_max,0,np.max(n)+np.max(n)/8,colors="r")
    p_value = (null_max > bench_max).sum() / null_repeats
    ax[1].annotate(f'Observed max ALE \n p-value = {p_value}', xy=(bench_max, np.max(n)+np.max(n)/8), ha='center')

    p05 = np.percentile(null_max, 95)
    line1 = ax[1].vlines(p05, 0, np.max(n), colors="darkgreen", label="p < 0.05")

    p01 = np.percentile(null_max, 99)
    line2 = ax[1].vlines(p01, 0, np.max(n), colors="green", label="p < 0.01")

    p001 = np.percentile(null_max, 99.9)
    line3 = ax[1].vlines(p001, 0, np.max(n), colors="lime", label="p < 0.001")

    ax[1].yaxis.tick_right()
    ax[1].set(xlabel="max ALE in roi mask")
    ax[1].title.set_text(f"{mask_name}_{index} - max ALE")
    ax[1].legend(handles=[line1, line2, line3], loc='upper right')
    
    fig.tight_layout()

    
    
    cwd = os.getcwd()
    if index is not None:
        if len(exp_names) == 2:
            fig.savefig(f"{cwd}/Results/Contrasts/ROI/Plots/{exp_names[0]}_x_{exp_names[1]}_{mask_name}_{index}")
        else:
            fig.savefig(f"{cwd}/Results/MainEffect/ROI/Plots/{exp_names}_{mask_name}_{index}")
    else:
        if len(exp_names) == 2:
            fig.savefig(f"{cwd}/Results/Contrasts/ROI/Plots/{exp_names[0]}_x_{exp_names[1]}_{mask_name}")
        else:
            fig.savefig(f"{cwd}/Results/MainEffect/ROI/Plots/{exp_names}_{mask_name}")
            
    plt.close()