import os
import pandas as pd
import numpy as np
import nibabel as nb
import pyarrow as pa
from utils.kernel import kernel_calc, kernel_conv
from utils.tal2icbm_spm import tal2icbm_spm
from utils.template import shape, affine

def read_exp_excel(input_path):
    
    df = pd.read_excel(input_path, engine='openpyxl')
    df = df[df.iloc[:,0].notnull()].reset_index(drop=True)

    lines_columns = ['Author', 'Subjects', 'XYZmm', 'Space', 'Cond', 'ExpIndex']
    lines = pd.DataFrame(columns=lines_columns)
    lines.Author = df.iloc[:, 0]
    lines.Subjects = df.iloc[:,1].astype(int)
    lines.XYZmm = [[df.iloc[:,2][i], df.iloc[:,3][i], df.iloc[:,4][i]]for i in range(df.shape[0])]
    lines.Space = df.iloc[:,5]
    lines.Cond = [df.iloc[i,6:].dropna().str.lower().str.strip().values for i in range(df.shape[0])]

    cnt_exp = 0
    first_line_idxs = [0]
    for i in range(lines.shape[0]):
        if i > 0:
            cnt_exp += 1
            if (lines.loc[i, ['Author', 'Subjects']] == lines.loc[i-1, ['Author', 'Subjects']]).all():
                if set(lines.at[i, 'Cond']) == set(lines.at[i-1, 'Cond']):
                    cnt_exp -= 1
                else:
                    first_line_idxs.append(i)
            else:
                first_line_idxs.append(i)
        lines.at[i, 'ExpIndex'] = cnt_exp
    num_exp = cnt_exp + 1
    
    return lines, first_line_idxs, num_exp

def create_exp_df(lines, first_line_idxs, num_exp):


    exp_columns = ['Author', 'Subjects', 'Space', 'Cond', 'XYZmm', 'UncertainTemplates',
                   'UncertainSubjects', 'Smoothing', 'XYZ', 'Kernels', 'MA', 'Peaks']
    exp_df = pd.DataFrame(columns=exp_columns)

    exp_df.Author = lines.Author[first_line_idxs]
    exp_df.Subjects = lines.Subjects[first_line_idxs]
    exp_df.Space = lines.Space[first_line_idxs]
    exp_df.Cond = lines.Cond[first_line_idxs]
    exp_df.XYZmm = [np.vstack(lines.loc[lines.ExpIndex == i, "XYZmm"]) for i in range(num_exp)]
    exp_df.loc[exp_df.Space == "TAL", "XYZmm"]  = exp_df[exp_df.Space == "TAL"].apply(lambda row: tal2icbm_spm(row.XYZmm), axis=1)
    exp_df.UncertainTemplates = 5.7/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))
    exp_df.UncertainSubjects = (11.6/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))) / np.sqrt(exp_df.Subjects)
    exp_df.Smoothing = np.sqrt(exp_df.UncertainTemplates**2 + exp_df.UncertainSubjects**2)
    padded_xyz = exp_df.apply(lambda row: np.pad(row.XYZmm, ((0,0),(0,1)), constant_values=[1]), axis=1).values
    exp_df.XYZ = [np.ceil(np.dot(np.linalg.inv(affine), xyzmm.T))[:3].astype(int) for xyzmm in padded_xyz]
    for i in range(num_exp):
        exp_df.XYZ.iloc[i][0][exp_df.XYZ.iloc[i][0] >= shape[0]] = shape[0] - 1
        exp_df.XYZ.iloc[i][1][exp_df.XYZ.iloc[i][1] >= shape[1]] = shape[1] - 1
        exp_df.XYZ.iloc[i][2][exp_df.XYZ.iloc[i][2] >= shape[2]] = shape[2] - 1
        exp_df.XYZ.iloc[i][exp_df.XYZ.iloc[i] < 1] = 1    
    exp_df.Kernels = exp_df.apply(lambda row: kernel_calc(affine, row.Smoothing, 31), axis=1)
    exp_df.MA = exp_df.apply(lambda row: kernel_conv(row.XYZ.T, row.Kernels), axis=1)
    exp_df.Peaks = [exp_df.XYZ.iloc[i].shape[1] for i in range(num_exp)]
    exp_df = exp_df.reset_index(drop=True)
    
    return exp_df

def create_task_df(exp_df):

    task_names, task_counts = np.unique(np.hstack(exp_df.Cond), return_counts=True)

    task_df_columns = ['Name', 'Num_Exp', 'Who', 'TotalSubjects', 'ExpIndex']
    task_df = pd.DataFrame(columns=task_df_columns)
    task_df.Name = np.append(task_names, 'all')
    task_df.Num_Exp = np.append(task_counts, exp_df.shape[0])

    for task_row, value in enumerate(list(task_df.Name)):
        counter = 0
        for exp_row in range(exp_df.shape[0]):
            if value in exp_df.at[exp_row, 'Cond']:
                if counter == 0:
                    task_df.at[task_row, 'Who'] = [exp_df.at[exp_row, 'Author']]
                    task_df.at[task_row, 'TotalSubjects'] = exp_df.at[exp_row, 'Subjects']
                    task_df.at[task_row, 'ExpIndex'] = [exp_row]
                else:
                    task_df.at[task_row, 'Who'].append(exp_df.at[exp_row, 'Author'])
                    task_df.at[task_row, 'TotalSubjects'] += exp_df.at[exp_row, 'Subjects']
                    task_df.at[task_row, 'ExpIndex'].append(exp_row)
                counter += 1


    task_df.at[task_df.index[-1], 'Who'] = exp_df.Author.to_list()
    task_df.at[task_df.index[-1], 'TotalSubjects'] = sum(exp_df.Subjects.to_list())
    task_df.at[task_df.index[-1], 'ExpIndex'] = list(range(exp_df.shape[0]))

    task_df = task_df.sort_values(by='Num_Exp', ascending=False).reset_index(drop=True)

    return task_df