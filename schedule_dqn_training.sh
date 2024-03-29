#!/bin/bash

#SBATCH --mem-per-cpu=3800
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -j train_ddqn_speederbikesim
#SBATCH -o train_output.txt

module load .gcc/11.2/openmpi/4.1.6
module load python/3.10.10

pip install numpy==1.25.1 pandas==2.0.3 gymnasium==0.29.1 pygame==2.5.2 scipy==1.12.0
# maually install torch for cuda 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# install custom simulation environment
pip install -e speederbikes-sim/
