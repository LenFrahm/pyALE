import nibabel as nb
from nilearn import datasets
import numpy as np
import os

template = datasets.load_mni152_brain_mask()

data = template.get_fdata()
shape = data.shape
pad_shape = [value+30 for value in shape]

prior = np.zeros(shape, dtype=bool)
prior[data > 0.1] = 1
sample_space = np.array(np.where(prior == 1))

affine = template.affine