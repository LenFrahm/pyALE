import argparse
import os
from os.path import isfile, isdir
from tkinter import filedialog
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

parser = argparse.ArgumentParser(description='Initates ALE Meta-Analysis.')
parser.add_argument('-f', metavar='folder', required=True, type=str, help='Folder that contains input and will be used for output')
parser.add_argument('-t', metavar='type', required=True, type=str, help='Type of Analysis; Possible types: M (MainEffect), P(Probabilistic), B(Balanced C), C(Full C)')
parser.add_argument('-n', metavar='name', required=True, type=str, help='Name given to analysis; used for provenance tracking')
parser.add_argument('-c', metavar='categories', type=str, nargs='+', default="All", help='Categories of experiments to include in analysis; check instructions file for syntax')
parser.add_argument('--n2', metavar='name2', type=str, help='If type is C or B a second analysis needs to be specified for the contrast')
parser.add_argument('--c2', metavar='categories2', type=str, nargs='+', help='Categories to include in 2nd analysis')



args = parser.parse_args()

folder = args.f
type_ = args.t
name = args.n
cats = args.c

if type_ == "C" or type_[0] == "B":
    name2 = args.n2
    cats2 = args.c2
    file2 = args.f2

print(cats)
"""
os.chdir(folder)
cwd = os.getcwd()

if not isdir("Results"):
    folder_setup(cwd)
    
if isfile(f'{cwd}/Results/experiments.pickle'):
    with open(f'{cwd}/Results/experiments.pickle', 'rb') as f:
        exp_all, tasks = pickle.load(f)
else:
    exp_all, tasks = read_exp_info(f'{cwd}/exp_info.xlsx')
    

if type_ == "M":
    exp_idxs, masks, mask_names = compile_studies(meta_df, row_idx, tasks)
    exp_df = exp_all.loc[exp_idxs].reset_index(drop=True)
    if len(exp_idxs) >= 12: 
        print(f'{exp_name} : {len(exp_idxs)} experiments; average of {exp_df.Subjects.mean():.2f} subjects per experiment')
        main_effect(exp_df, exp_name, cluster_thresh=0.001, null_repeats=1000)
        contribution(exp_df, exp_name, exp_idxs, tasks)
    else:
        print(f"{exp_name} : only {len(exp_idxs)} experiments - not analyzed!")

    if len(masks) > 0:
        print(f"{exp_name} - ROI analysis")
        with open(f"Results/MainEffect/Full/NullDistributions/{exp_name}_null.pickle", 'rb') as f:
            null_ale, _, _, _ = pickle.load(f) 
        null_ale = np.stack(null_ale)
        check_rois(exp_df, exp_name, masks, mask_names, null_repeats=1000, null_ale=null_ale)

"""