#!/bin/sh
#SBATCH --account=jinm72
#SBATCH --mail-user=l.frahm@fz-juelich.de
#SBATCH --mail-type=END
#SBATCH --job-name=sim_main
#SBATCH --time=0:20:00
#SBATCH --output=/p/scratch/cinm-7/pyALE_TFCE/output/logs/main-out.%j
#SBATCH --error=/p/scratch/cinm-7/pyALE_TFCE/output/logs/main-err.%j
#SBATCH --ntasks=256
#SBATCH --exclusive
#SBATCH --partition=dc-cpu-devel

module load GCC/9.3.0
module load OpenMPI/4.1.0rc1
module load mpi4py
module load GCCcore/.9.3.0
module load Python/3.8.5
module load Python-Neuroimaging/2020-Python-3.8.5
module load parallel


# the --exclusive to srun makes srun use distinct CPUs for each job step
# -N1 -n1 allocates a single core to each task
srun="srun --exclusive -N1 -n1"

# --delay .2 prevents overloading the controlling node
# -j is the number of tasks parallel runs so we set it to $SLURM_NTASKS
parallel="parallel --delay .2 -j $SLURM_NTASKS"

# this runs the parallel command we want
$parallel "$srun python me.py {1} {2} {3}" ::: {19..20} ::: {8..9} ::: {0..499}

wait

python me_2_hdf5.py 19 20 8 9