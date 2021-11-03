import pandas as pd
import nibabel as nb
import numpy as np

def compile_studies(conditions, tasks):
    to_use = []
    not_to_use = []
    masks = []
    mask_names = []

    for condition in conditions:
        operation = condition[0]

        if operation == "+":
            tag = condition[1:].lower()
            to_use.append(tasks[tasks.Name == tag].ExpIndex.to_list()[0])

        if operation == "-":
            tag = condition[1:].lower()
            not_to_use.append(tasks[tasks.Name == tag].ExpIndex.to_list()[0])

        if operation == "?":
            tag = condition[1:].lower()
            flat_list = [number for sublist in to_use for number in sublist]
            to_use = []
            to_use.append(list(set(flat_list)))
            
        if operation == '$':
            mask_file = condition[1:]
            mask = nb.load(mask_file).get_fdata()
            if np.unique(mask).shape[0] == 2:
                #binary mask
                masks.append(mask.astype(bool))
            else:
                masks.append(mask.astype(int))
            mask_names.append(mask_file[:-4])
            
            
    use_sets = map(set, to_use)
    to_use = list(set.intersection(*use_sets))
    
    if len(not_to_use) > 0:
        not_use_sets = map(set, not_to_use)
        not_to_use = list(set.intersection(*not_use_sets))
        
        to_use = to_use.differences(not_to_use)
        
    return to_use, masks, mask_names