#!/bin/bash
#SBATCH -J w_1gpu         # Your job name to be displayed by squeue
#SBATCH -o /usr/users/roudi/whisper-flamingo/slurm/train_audio_slurm/ft_w_1gpu_%j.out   # path to write stdout, %j will be jobID
#SBATCH -e /usr/users/roudi/whisper-flamingo/slurm/train_audio_slurm/ft_w_1gpu_%j.err   # path to write stdout, %j will be jobIDD
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=a6
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1 # assert ntasks_per_node == cfg.distributed_world_size // nnodes

# print hostname
srun hostname

## Set the python environment you want to use for your code
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/usr/users/roudi/vtenvs/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd /usr/users/roudi/whisper-flamingo

# srun python -u whisper_ft_muavic.py config/audio/audio_en_large.yaml
srun python -u whisper_ft_muavic.py config/audio/audio_lrs2_medium.yaml
