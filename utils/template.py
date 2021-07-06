import nibabel as nb
import numpy as np
import os

path = os.path.abspath(__file__)
template = nb.load(f"{path[:-11]}/mask/Grey10.nii")

data = template.get_fdata()
shape = data.shape
pad_shape = np.array([value+30 for value in shape])

prior = np.zeros(shape, dtype=bool)
prior[data > 0.1] = 1
sample_space = np.array(np.where(prior == 1))

affine = template.affine