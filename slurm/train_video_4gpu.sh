#!/bin/bash
#SBATCH -J wftv_4gpu         # Your job name to be displayed by squeue
#SBATCH -o /usr/users/roudi/whisper-flamingo/slurm/train_video_slurm/vsr_4gpu_%j.out   # path to write stdout, %j will be jobID
#SBATCH -e /usr/users/roudi/whisper-flamingo/slurm/train_video_slurm/vsr_4gpu_%j.err
#SBATCH --qos=priority
##SBATCH --qos=regular
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --partition=a6
##SBATCH --partition=a5
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4 # assert ntasks_per_node == cfg.distributed_world_size // nnodes

# print hostname
srun hostname

## Set the python environment you want to use for your code
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/usr/users/roudi/vtenvs/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd /usr/users/roudi/whisper-flamingo

# srun python -u whisper_ft_muavic_video.py config/audio-visual/av_en_large.yaml
srun python -u whisper_ft_muavic_video.py config/audio-visual/av_multi_small.yaml
# srun python -u whisper_ft_muavic_video.py config/audio-visual/av_multi_medium.yaml
