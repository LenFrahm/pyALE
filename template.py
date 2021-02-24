import nibabel as nb
import numpy as np
import os

cwd = os.getcwd()

template = nb.load(f"{cwd}/MaskenEtc/Grey10.nii")
data = template.get_fdata()
shape = data.shape
prior = np.zeros(shape, dtype=bool)
prior[data > 0.1] = 1
affine = template.affine
pad_shape = [value+30 for value in shape]
sample_space = np.array(np.where(prior == 1))