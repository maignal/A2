#!/usr/bin/env bash
#SBATCH --account cs-302
#SBATCH --qos cs-302
#SBATCH --partition academic
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 32
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --time 00:30:00

module purge
module load gcc
module load openmpi

echo STARTING AT `date`

# compile
make

# executes
# srun ./prog0 10 1000000
# srun ./progA 10 1000000
# srun ./progB 10 1000000 25000 25000
# srun ./progC 10 1000000 25000 25000
# srun ./progD 10 1000000

for i in 100 1000 10000 100000; do
    srun -n 1 ./prog0 10 $i
done

echo FINISHED at `date`
