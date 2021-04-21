import h5py
import numpy as np
import sys
import shutil
import os

if __name__ == "__main__":
    ns_min = int(sys.argv[1])
    ns_max = int(sys.argv[2])
    nt_min = int(sys.argv[3])
    nt_max = int(sys.argv[4])


    with h5py.File("/p/scratch/cinm-7/pyALE_TFCE/output/main.hdf5", "a") as f:
        for ns in range(ns_min,ns_max+1):
            for nt in range(nt_min,nt_max+1):
                for rep in range(0,500):
                    npz = np.load(f"/p/scratch/cinm-7/pyALE_TFCE/output/main/{ns}_{nt}_{rep}.npz")
                    f.create_dataset(f"hx_conv/{ns}/{nt}/{rep}", data=npz["hx_conv"], compression="gzip", compression_opts=5)
                    f.create_dataset(f"kernels/{ns}/{nt}/{rep}", data=npz["kernels"], compression="gzip", compression_opts=5)
                    f.create_dataset(f"num_foci/{ns}/{nt}/{rep}", data=npz["num_foci"], compression="gzip", compression_opts=5)
                    f.create_dataset(f"ale/{ns}/{nt}/{rep}", data=npz["ale"], compression="gzip", compression_opts=5)
                    f.create_dataset(f"z/{ns}/{nt}/{rep}", data=npz["z"], compression="gzip", compression_opts=5)
                    f.create_dataset(f"tfce/{ns}/{nt}/{rep}", data=npz["tfce"], compression="gzip", compression_opts=5)
                    
    
                    
    shutil.rmtree("/p/scratch/cinm-7/pyALE_TFCE/output/main")
    os.mkdir("/p/scratch/cinm-7/pyALE_TFCE/output/main")