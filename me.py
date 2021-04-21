import os
import sys
from os.path import isfile, isdir
import pandas as pd
import numpy as np
import h5py
from utils.template import sample_space, affine
from utils.kernel import kernel_calc
from utils.compute import compute_ma, compute_hx, compute_hx_conv, compute_ale, compute_z, compute_tfce

if __name__ == '__main__':
    num_studies = int(sys.argv[1])
    num_true = int(sys.argv[2])
    rep = int(sys.argv[3])

    with h5py.File(f"/p/scratch/cinm-7/pyALE_TFCE/inputs/simulation_data.hdf5", "r") as sim_data:
        sample_sizes = sim_data[f"{num_studies}/{num_true}/{rep}/sample_sizes"][:]
        foci = []
        for key in sim_data[f"{num_studies}/{num_true}/{rep}/foci"].keys():
            foci.append(sim_data[f"{num_studies}/{num_true}/{rep}/foci"][key][:])
        
            num_foci = [foc.shape[0] for foc in foci]

    kernels = []
    for sample_size in sample_sizes:
        temp_uncertainty = 5.7/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))
        subj_uncertainty = (11.6/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))) / np.sqrt(sample_size)
        smoothing = np.sqrt(temp_uncertainty**2 + subj_uncertainty**2)
        kernels.append(kernel_calc(affine, smoothing, 31))

    mb = 1
    for i in range(num_studies):
        mb = mb*(1-np.max(kernels[i]))

    # define bins for histogram
    bin_steps=0.0001
    bin_edges = np.arange(0.00005,1-mb+0.001,bin_steps)
    bin_centers = np.arange(0,1-mb+0.001,bin_steps)
    step = int(1/bin_steps)

    ma = compute_ma(foci, kernels)
    hx = compute_hx(ma, bin_edges)
    ale = compute_ale(ma)
    hx_conv = compute_hx_conv(hx, bin_centers, step)
    z = compute_z(ale, hx_conv, step)
    tfce = compute_tfce(z)
    
    
    np.savez_compressed(f"/p/scratch/cinm-7/pyALE_TFCE/output/main/{num_studies}_{num_true}_{rep}", hx_conv=hx_conv, kernels=kernels, num_foci=num_foci, ale=ale, z=z, tfce=tfce)
    
    """ with h5py.File("/p/scratch/cinm-7/pyALE_TFCE/output/main.hdf5", driver='mpio', comm=MPI.COMM_WORLD) as f:
        f.create_dataset(f"hx_conv/{num_studies}/{num_true}/{rep}", data=hx_conv)
        f.create_dataset(f"kernels/{num_studies}/{num_true}/{rep}", data=kernels)
        f.create_dataset(f"num_foci/{num_studies}/{num_true}/{rep}", data=num_foci)
        f.create_dataset(f"ale/{num_studies}/{num_true}/{rep}", data=ale)
        f.create_dataset(f"z/{num_studies}/{num_true}/{rep}", data=z)
        f.create_dataset(f"tfce/{num_studies}/{num_true}/{rep}", data=tfce)"""