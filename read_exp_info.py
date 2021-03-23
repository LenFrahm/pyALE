import os
import pandas as pd
import numpy as np
import nibabel as nb
import pickle
from kernel import kernel_calc, kernel_conv
from tal2icbm_spm import tal2icbm_spm
from template import shape, affine


def read_exp_info(filename):

    df = pd.read_excel(filename, engine='openpyxl')
    df = df[df['Articles'].notnull()].reset_index(drop=True)

    x_max = shape[0]
    y_max = shape[1]
    z_max = shape[2]

    lines_columns = ['Author', 'Subjects', 'XYZmm', 'Space', 'Cond', 'ExpIndex']
    lines = pd.DataFrame(columns=lines_columns)
    lines.Author = df.iloc[:, 0]
    lines.Subjects = df.iloc[:,1].astype(int)
    lines.XYZmm = [[df.iloc[:,2][i], df.iloc[:,3][i], df.iloc[:,4][i]]for i in range(df.shape[0])]
    lines.Space = df.iloc[:,5]
    lines.Cond = [df.iloc[i,6:].dropna().str.lower().str.strip().values for i in range(df.shape[0])]

    cnt_exp = 0
    first_lines = [0]
    for i in range(lines.shape[0]):
        if i > 0:
            cnt_exp += 1
            if (lines.loc[i, ['Author', 'Subjects']] == lines.loc[i-1, ['Author', 'Subjects']]).all():
                if set(lines.at[i, 'Cond']) == set(lines.at[i-1, 'Cond']):
                    cnt_exp -= 1
                else:
                    first_lines.append(i)
            else:
                first_lines.append(i)
        lines.at[i, 'ExpIndex'] = cnt_exp

    num_exp = cnt_exp + 1

    exp_columns = ['Author', 'Subjects', 'Space', 'Cond', 'XYZmm', 'UncertainTemplates',
                   'UncertainSubjects', 'Smoothing', 'XYZ', 'Kernels', 'MA', 'Peaks']
    experiments = pd.DataFrame(columns=exp_columns)

    experiments.Author = lines.Author[first_lines]
    experiments.Subjects = lines.Subjects[first_lines]
    experiments.Space = lines.Space[first_lines]
    experiments.Cond = lines.Cond[first_lines]
    experiments.XYZmm = [np.vstack(lines.loc[lines.ExpIndex == i, "XYZmm"]) for i in range(num_exp)]
    experiments.loc[experiments.Space == "TAL", "XYZmm"]  = experiments[experiments.Space == "TAL"].apply(lambda row: tal2icbm_spm(row.XYZmm), axis=1)
    experiments.UncertainTemplates = 5.7/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))
    experiments.UncertainSubjects = (11.6/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))) / np.sqrt(experiments.Subjects)
    experiments.Smoothing = np.sqrt(experiments.UncertainTemplates**2 + experiments.UncertainSubjects**2)

    padded_xyz = experiments.apply(lambda row: np.pad(row.XYZmm, ((0,0),(0,1)), constant_values=[1]), axis=1).values
    experiments.XYZ = [np.ceil(np.dot(np.linalg.inv(affine), xyzmm.T))[:3].astype(int) for xyzmm in padded_xyz]
    for i in range(num_exp):
        experiments.XYZ.iloc[i][0][experiments.XYZ.iloc[i][0] >= x_max] = x_max-1
        experiments.XYZ.iloc[i][1][experiments.XYZ.iloc[i][1] >= y_max] = y_max-1
        experiments.XYZ.iloc[i][2][experiments.XYZ.iloc[i][2] >= z_max] = z_max-1    
        experiments.XYZ.iloc[i][experiments.XYZ.iloc[i] < 1] = 1

    experiments.Kernels = experiments.apply(lambda row: kernel_calc(affine, row.Smoothing, 31), axis=1)
    experiments.MA = experiments.apply(lambda row: kernel_conv(row.XYZ.T, row.Kernels), axis=1)
    experiments.Peaks = [experiments.XYZ.iloc[i].shape[1] for i in range(num_exp)]
    experiments = experiments.reset_index(drop=True)

    task_names, task_counts = np.unique(np.hstack(experiments.Cond), return_counts=True)

    tasks_columns = ['Name', 'Num_Exp', 'Who', 'TotalSubjects', 'ExpIndex']
    tasks = pd.DataFrame(columns=tasks_columns)
    tasks.Name = np.append(task_names, 'all')
    tasks.Num_Exp = np.append(task_counts, experiments.shape[0])

    for task_row, value in enumerate(list(tasks.Name)):
        counter = 0
        for exp_row in range(lines.iloc[-1]['ExpIndex'] + 1):
            if value in experiments.at[exp_row, 'Cond']:
                if counter == 0:
                    tasks.at[task_row, 'Who'] = [experiments.at[exp_row, 'Author']]
                    tasks.at[task_row, 'TotalSubjects'] = experiments.at[exp_row, 'Subjects']
                    tasks.at[task_row, 'ExpIndex'] = [exp_row]
                else:
                    tasks.at[task_row, 'Who'].append(experiments.at[exp_row, 'Author'])
                    tasks.at[task_row, 'TotalSubjects'] += experiments.at[exp_row, 'Subjects']
                    tasks.at[task_row, 'ExpIndex'].append(exp_row)
                counter += 1


    tasks.at[tasks.index[-1], 'Who'] = experiments.Author.to_list()
    tasks.at[tasks.index[-1], 'TotalSubjects'] = sum(experiments.Subjects.to_list())
    tasks.at[tasks.index[-1], 'ExpIndex'] = list(range(experiments.shape[0]))

    tasks = tasks.sort_values(by='Num_Exp', ascending=False).reset_index(drop=True)



    cwd = os.getcwd()
    pickle_object = (experiments, tasks)
    with open(f'{cwd}/Results/{filename}.pickle', 'wb') as f:
        pickle.dump(pickle_object, f)
        
    return experiments, tasks