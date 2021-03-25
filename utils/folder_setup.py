import os

def folder_setup(path, analysis):

    #MainEffect
    if analysis == "MainEffect_Full":
        basepath = f"{path}/Results/MainEffect/Full"
        os.makedirs(f"{basepath}/Volumes/Foci")
        os.makedirs(f"{basepath}/Volumes/ALE")
        os.makedirs(f"{basepath}/Volumes/Z")
        os.makedirs(f"{basepath}/Volumes/TFCE")
        os.makedirs(f"{basepath}/Volumes/Corrected")
        os.makedirs(f"{basepath}/Images/Foci")
        os.makedirs(f"{basepath}/Images/ALE")
        os.makedirs(f"{basepath}/Images/TFCE")
        os.makedirs(f"{basepath}/Images/Corrected")        
        os.makedirs(f"{basepath}/Contribution/")
        os.makedirs(f"{basepath}/NullDistributions")
        
    if analysis == "MainEffect_CV":
        basepath = f"{path}/Results/MainEffect/CV"
        os.makedirs(f"{basepath}/Volumes")
        os.makedirs(f"{basepath}/Images/")
        os.makedirs(f"{basepath}/NullDistributions")

    if analysis == "MainEffect_ROI":
        basepath = f"{path}/Results/MainEffect/ROI"
        os.makedirs(f"{basepath}/Plots")
        os.makedirs(f"{basepath}/NullDistributions")
    
    #Contrast
    if analysis == "Contrast_Full":
        basepath = f"{path}/Results/Contrast/Full"
        os.makedirs(f"{basepath}/Images")
        os.makedirs(f"{basepath}/NullDistributions")
        os.makedirs(f"{basepath}/Conjunctions/Images")

    if analysis == "Contrast_Balanced":
        basepath = f"{path}/Results/Contrast/Balanced"
        os.makedirs(f"{basepath}/Images")
        os.makedirs(f"{basepath}/NullDistributions")
        os.makedirs(f"{basepath}/Conjunctions/Images")

    if analysis == "Contrast_ROI":
        basepath = f"{path}/Results/Contrast/ROI"
        os.makedirs(f"{basepath}/Plots")
        os.makedirs(f"{basepath}/NullDistributions")