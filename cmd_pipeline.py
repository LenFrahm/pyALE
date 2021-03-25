import argparse
import pickle
import os
from os.path import isfile, isdir
import pandas as pd
import numpy as np
from analysis.main_effect import main_effect
from analysis.contrast import contrast
from analysis.legacy_contrast import legacy_contrast
from analysis.roi import check_rois
from utils.compile_studies import compile_studies
from utils.contribution import contribution
from utils.folder_setup import folder_setup
from utils.read_exp_info import read_exp_info

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Initates ALE Meta-Analysis.')
    parser.add_argument('path', type=str, help='Path to folder that contains excel sheets and will be used to store output.')
    parser.add_argument('file', type=str, help='Filename of excel file that stores study information (coordinates, etc.)')
    parser.add_argument('type', type=str, help='Type of Analysis; check instructions file for possible types')
    parser.add_argument('name', type=str, help='Name given to analysis; used for provenance tracking')
    parser.add_argument('tags', type=str, nargs='+', default="All", help='Tags of experiments to include in analysis; check instructions file for syntax')
    parser.add_argument('-n2', metavar='name2', type=str, help='for contrasts there needs to be a 2nd analysis specified; name of the second')
    parser.add_argument('-t2', metavar='tags', type=str, nargs='+', help='categories for 2nd analysis')
    parser.add_argument('-nr', metavar='n_rep', type=int, help='number of iterations to use for null simulation')
    parser.add_argument('-ct', metavar='cluster_thresh', type=int, help='significance level used to threshold cluster forming')
    parser.add_argument('-sn', metavar='sample_n', type=int, help='amount of subsamples taken from experiment pool; used in P and B')
    parser.add_argument('-dt', metavar='diff_thresh', type=int, help='threshold used to compare contrast differences against')
    parser.add_argument('-dr', metavar='diff_rep', type=int, help='amount of sampling repititions used when comparing to subsampled ALEs in a balanced Contrast')
    args = parser.parse_args()
    
    path = args.path
    os.chdir(path)
    file = args.file
    type_ = args.type
    exp_name = args.name
    conditions = args.tags
    if type_[0] in ["C", "B"]:
        if args.n2 is None:
            print("When selecting one of the contrast procedures a 2nd analysis needs to be specified. Please check --help for syntax specifics.")
            exit()
        else:
            exp_name2 = args.n2
            conditions2 = args.t2
    
    if args.nr is not None:
        n_rep = args.nr
    else:
        n_rep = 1000
        
    if args.ct is not None:
        cluster_thresh = args.ct
    else:
        cluster_thresh = 0.001
    
    if args.sn is not None:
        sample_n = args.sn
    else:
        sample_n = 2500
    
    if args.dt is not None:
        diff_thresh = args.dt
    else:
        diff_thresh = 0.05
        
    if args.dr is not None:
        diff_repeats = args.dr
    else:
        diff_repeats = 5000


    if not isdir("Results"):
        os.mkdir("Results")

    if isfile(f'Results/{file}.pickle'):
        with open(f'Results/{file}.pickle', 'rb') as f:
            exp_all, tasks = pickle.load(f)
    else:
        exp_all, tasks = read_exp_info(f'{file}')

    if type_ == "M":
        if not isdir("Results/MainEffect/Full"):
            folder_setup(path, "MainEffect_Full")
        exp_idxs, masks, mask_names = compile_studies(conditions, tasks)
        exp_df = exp_all.loc[exp_idxs].reset_index(drop=True)
        if len(exp_idxs) >= 12: 
            print(f'{exp_name} : {len(exp_idxs)} experiments; average of {exp_df.Subjects.mean():.2f} subjects per experiment')
            main_effect(exp_df, exp_name, cluster_thresh=cluster_thresh, null_repeats=n_rep)
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
            check_rois(exp_df, exp_name, masks, mask_names, null_repeats=n_rep, null_ale=null_ale)

    if type_[0] == "P":
        if not isdir("Results/MainEffect/CV"):
            folder_setup(path, "MainEffect_CV")
        exp_idxs, _, _ = compile_studies(conditions, tasks)
        exp_df = exp_all.loc[exp_idxs].reset_index(drop=True)
        if len(type_) > 1:
            target_n = type_[1:]
            main_effect(exp_df, exp_name, null_repeats=n_rep, target_n=target_n, sample_n=sample_n)
        else:
            print(f"{exp_name}: need to specify subsampling")

    if type_ == "C":
        if not isdir("Results/Contrast/Full"):
            folder_setup(path, "Contrast_Full")
        if not isdir("Results/MainEffect/Full"):
            folder_setup(path, "MainEffect_Full") 
        exp_idx1, masks, mask_names = compile_studies(conditions, tasks)
        exp_idxs = [exp_idx1, compile_studies(conditions2, tasks)[0]]
        exp_dfs = [exp_all.loc[exp_idxs[0]].reset_index(drop=True), exp_all.loc[exp_idxs[1]].reset_index(drop=True)]
        if len(exp_idxs[0]) >= 12 and len(exp_idxs[1]) >= 12:
            if not isfile(f"Results/MainEffect/Full/Volumes/Corrected/{exp_name}_cFWE05.nii"):
                main_effect(exp_dfs[0], exp_name, null_repeats = n_rep)
                contribution(exp_dfs[0], exp_name, exp_idxs[0], tasks)
            if not isfile(f"Results/MainEffect/Full/Volumes/Corrected/{exp_name2}_cFWE05.nii"):
                main_effect(exp_dfs[1], exp_name2, null_repeats = n_rep)
                contribution(exp_dfs[1], exp_name2, exp_idxs[1], tasks)

            for i in reversed(exp_idxs[0]):
                if i in exp_idxs[1]:
                    exp_idxs[0].remove(i)
                    exp_idxs[1].remove(i)

            exp_dfs = [exp_all.loc[exp_idxs[0]].reset_index(drop=True), exp_all.loc[exp_idxs[1]].reset_index(drop=True)]
            legacy_contrast(exp_dfs, [exp_name, exp_name2], diff_thresh=diff_thresh, null_repeats=n_rep)

            if len(masks) > 0:
                print(f"{exp_name} x {exp_name2} - ROI analysis")
                if not isdir("Results/Contrast/ROI"):
                    folder_setup(path, "Contrast_ROI")   
                check_rois(exp_dfs, exp_names, masks, mask_names, null_repeats=n_rep)

    if type_[0] == "B":
        if not isdir("Results/Contrast/Balanced"):
            folder_setup(path, "Contrast_Balanced")
        if not isdir("Results/MainEffect/CV"):
            folder_setup(path, "MainEffect_CV")
        exp_idx1, _, _ = compile_studies(conditions, tasks)
        exp_idxs = [exp_idx1, compile_studies(conditions2, tasks)[0]]
        exp_dfs = [exp_all.loc[exp_idxs[0]].reset_index(drop=True), exp_all.loc[exp_idxs[1]].reset_index(drop=True)]
        n = [len(exp_idxs[0]), len(exp_idxs[1])]

        if np.min(n) >= 19:

            if len(type_) > 1:
                target_n = int(type_[1:])

            else:
                target_n = int(min(np.floor(np.mean((np.min(n), 17))), np.min(n)-2))

            if not isfile(f'Results/MainEffect/CV/Volumes/{exp_name}_{target_n}.nii'):
                main_effect(exp_dfs[0], exp_name, null_repeats=n_rep, target_n=target_n, sample_n=sample_n)
            if not isfile(f'Results/MainEffect/CV/Volumes/{exp_name2}_{target_n}.nii'):
                main_effect(exp_dfs[1], exp_name2, null_repeats=n_rep, target_n=target_n, sample_n=sample_n)
            
            contrast(exp_dfs, [exp_name, exp_name2], null_repeats=n_rep, target_n=target_n, diff_repeats=diff_rep)