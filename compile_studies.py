import pandas as pd

def compile_studies(exp_df, row_index, tasks):
    conditions = exp_df.iloc[row_index, 4:].dropna().to_list()
    to_use = []
    not_to_use = []
    for condition in conditions:
        operation = condition[0]
        tag = condition[1:].lower()

        if operation == "+":
            to_use.append(tasks[tasks.Name == tag].ExpIndex.to_list()[0])

        if operation == "-":
            not_to_use.append(tasks[tasks.Name == tag].ExpIndex.to_list()[0])

        if operation == "?":
            flat_list = [number for sublist in to_use for number in sublist]
            to_use = []
            to_use.append(list(set(flat_list)))
            
        #if operation == '#':
        # Voi Analysis needs to be added at a later point
    use_sets = map(set, to_use)
    to_use = list(set.intersection(*use_sets))
    
    if len(not_to_use) > 0:
        not_use_sets = map(set, not_to_use)
        not_to_use = list(set.intersection(*not_use_sets))
        
        to_use = to_use.differences(not_to_use)
        
    return to_use