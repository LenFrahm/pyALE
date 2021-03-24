import os
from os.path import isfile, isdir
import pandas as pd
import numpy as np
import pickle
from main_effect import main_effect
from contrast import contrast
from legacy_contrast import legacy_contrast
from compile_studies import compile_studies
from contribution import contribution
from folder_setup import folder_setup
from roi import check_rois
from read_exp_info import read_exp_info

def setup(path, analysis_info_name, experiment_info_name):
    os.chdir(path)
    meta_df = pd.read_excel(f"{analysis_info_name}", engine='openpyxl', header=None)

    if not isdir("Results"):
        os.mkdir("Results")

    if isfile(f'Results/{experiment_info_name}.pickle'):
        with open(f'Results/{experiment_info_name}.pickle', 'rb') as f:
            exp_all, tasks = pickle.load(f)
    else:
        exp_all, tasks = read_exp_info(f'{experiment_info_name}')
        
    return meta_df, exp_all, tasks

def analysis(path, meta_df, exp_all, tasks, null_repeats=5000, cluster_thresh=0.001, sample_n=2500, diff_thresh=0.05, diff_repeats=1000):
    os.chdir(path)
    
    for row_idx in range(meta_df.shape[0]):
        if type(meta_df.iloc[row_idx, 0]) != str:
            continue
        if meta_df.iloc[row_idx, 0] == 'M': #Main Effect Analysis       
            if not isdir("Results/MainEffect/Full"):
                folder_setup(path, "MainEffect_Full")           
            exp_name = meta_df.iloc[row_idx, 1]
            conditions = meta_df.iloc[row_idx, 2:].dropna().to_list()
            exp_idxs, masks, mask_names = compile_studies(conditions, tasks)
            exp_df = exp_all.loc[exp_idxs].reset_index(drop=True)
            if len(exp_idxs) >= 12: 
                print(f'{exp_name} : {len(exp_idxs)} experiments; average of {exp_df.Subjects.mean():.2f} subjects per experiment')
                main_effect(exp_df, exp_name, cluster_thresh=0.001, null_repeats=null_repeats)
                contribution(exp_df, exp_name, exp_idxs, tasks)
            else:
                print(f"{exp_name} : only {len(exp_idxs)} experiments - not analyzed!")

            if len(masks) > 0:
                print(f"{exp_name} - ROI analysis")
                if not isdir("Results/MainEffect/ROI"):
                    folder_setup(path, "MainEffect_ROI")   
                with open(f"Results/MainEffect/Full/NullDistributions/{exp_name}_null.pickle", 'rb') as f:
                    null_ale, _, _, _ = pickle.load(f) 
                null_ale = np.stack(null_ale)
                check_rois(exp_df, exp_name, masks, mask_names, null_repeats=null_repeats, null_ale=null_ale)

        if meta_df.iloc[row_idx, 0][0] == "P": # Probabilistic ALE
            if not isdir("Results/MainEffect/CV"):
                folder_setup(path, "MainEffect_CV")   
            exp_name = meta_df.iloc[row_idx, 1]
            conditions = meta_df.iloc[row_idx, 2:].dropna().to_list()
            exp_idxs, _, _ = compile_studies(conditions, tasks)
            exp_df = exp_all.loc[exp_idxs].reset_index(drop=True)
            if len(meta_df.iloc[row_idx, 0]) > 1:
                target_n = int(meta_df.iloc[row_idx, 0][1:])
                main_effect(exp_df, exp_name, null_repeats=null_repeats, target_n=target_n, sample_n=sample_n)
            else:
                print(f"{exp_name}: need to specify subsampling")
                continue

        if meta_df.iloc[row_idx, 0] == 'C': # Contrast Analysis
            if not isdir("Results/Contrast/Full"):
                folder_setup(path, "Contrast_Full")   
            exp_names = [meta_df.iloc[row_idx, 1], meta_df.iloc[row_idx+1, 1]]
            conditions = [meta_df.iloc[row_idx, 2:].dropna().to_list(), meta_df.iloc[row_idx+1, 2:].dropna().to_list()]
            exp_idx1, masks, mask_names = compile_studies(conditions[0], tasks)
            exp_idxs = [exp_idx1, compile_studies(conditions[1], tasks)[0]]
            exp_dfs = [exp_all.loc[exp_idxs[0]].reset_index(drop=True), exp_all.loc[exp_idxs[1]].reset_index(drop=True)]

            if len(exp_idxs[0]) >= 12 and len(exp_idxs[1]) >= 12:
                if not isfile(f"Results/MainEffect/Full/Volumes/Corrected/{exp_names[0]}_cFWE05.nii"):
                    main_effect(exp_dfs[0], exp_names[0], null_repeats = null_repeats)
                    contribution(exp_dfs[0], exp_names[0], exp_idxs[0], tasks)
                if not isfile(f"Results/MainEffect/Full/Volumes/Corrected/{exp_names[1]}_cFWE05.nii"):
                    main_effect(exp_dfs[1], exp_names[1], null_repeats = null_repeats)
                    contribution(exp_dfs[1], exp_names[1], exp_idxs[1], tasks)

                for i in reversed(exp_idxs[0]):
                    if i in exp_idxs[1]:
                        exp_idxs[0].remove(i)
                        exp_idxs[1].remove(i)

                exp_dfs = [exp_all.loc[exp_idxs[0]].reset_index(drop=True), exp_all.loc[exp_idxs[1]].reset_index(drop=True)]

                legacy_contrast(exp_dfs, exp_names, diff_thresh=diff_thresh, null_repeats=null_repeats)

                if len(masks) > 0:
                    print(f"{exp_names[0]} x {exp_names[1]} - ROI analysis")
                    if not isdir("Results/Contrast/ROI"):
                        folder_setup(path, "Contrast_ROI")   
                    check_rois(exp_dfs, exp_names, masks, mask_names, null_repeats=null_repeats)

        if meta_df.iloc[row_idx, 0][0] == 'B': # Balanced Contrast Analysis:
            if not isdir("Results/Contrast/Balanced"):
                folder_setup(path, "Contrast_Balanced")   
            exp_names = [meta_df.iloc[row_idx, 1], meta_df.iloc[row_idx+1, 1]]
            conditions = [meta_df.iloc[row_idx, 2:].dropna().to_list(), meta_df.iloc[row_idx+1, 2:].dropna().to_list()]
            exp_idx1, _, _ = compile_studies(conditions[0], tasks)
            exp_idxs = [exp_idx1, compile_studies(conditions[1], tasks)[0]]
            exp_dfs = [exp_all.loc[exp_idxs[0]].reset_index(drop=True), exp_all.loc[exp_idxs[1]].reset_index(drop=True)]
            n = [len(exp_idxs[0]), len(exp_idxs[1])]

            if np.min(n) >= 19:

                if len(meta_df.iloc[row_idx, 0]) > 1:
                    target_n = int(meta_df.iloc[row_idx, 0][1:])

                else:
                    target_n = int(min(np.floor(np.mean((np.min(n), 17))), np.min(n)-2))

                if not isfile(f'Results/MainEffect/CV/Volumes/{exp_names[0]}_{target_n}.nii'):
                    main_effect(exp_dfs[0], exp_names[0], null_repeats=null_repeats, target_n=target_n, sample_n=sample_n)
                if not isfile(f'Results/MainEffect/CV/Volumes/{exp_names[1]}_{target_n}.nii'):
                    main_effect(exp_dfs[1], exp_names[1], null_repeats=null_repeats, target_n=target_n, sample_n=sample_n)

                contrast(exp_dfs, exp_names, null_repeats=null_repeats, target_n=target_n, diff_repeats=diff_repeats)