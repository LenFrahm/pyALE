import pandas as pd

def compile_studies(df, row_index, experiments, tasks):
    conditions = df.iloc[row_index, 4:].dropna().to_list()
    to_use = []
    for condition in conditions:
        operation = condition[0]
        tag = condition[1:].lower()

        if operation == "+":
            to_use.append(tasks[tasks.Name == tag].ExpIndex.to_list()[0])

        if operation == "-":
            not_to_use = tasks[tasks.Name == tag].ExpIndex.to_list()[0]

        if operation == "?":
            flat_list = [number for sublist in to_use for number in sublist]
            to_use = []
            to_use.append(list(set(flat_list)))
            
        #if operation == '#':
        # Voi Analysis needs to be added at a later point

    sets = map(set, to_use)
    to_use = list(set.intersection(*sets))
    return to_use