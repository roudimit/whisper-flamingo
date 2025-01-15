#!/bin/bash
#SBATCH -J decode_avsr          # Your job name to be displayed by squeue
#SBATCH -o /usr/users/roudi/whisper-flamingo/slurm/decode_slurm/whisper_video_%j.out   # path to write stdout, %j will be jobID
#SBATCH -e /usr/users/roudi/whisper-flamingo/slurm/decode_slurm/whisper_video_%j.err    # path to write stderr, %j will be jobID
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --requeue # restart if killed
#SBATCH --partition=a6,a5,2080
##SBATCH --partition=a5,a6,2080
#SBATCH --exclude sls-a6-5
#SBATCH --mem=22G
#SBATCH --ntasks-per-node=1 # assert ntasks_per_node == cfg.distributed_world_size // nnodes

## Set the python environment you want to use for your code
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/usr/users/roudi/vtenvs/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd /usr/users/roudi/whisper-flamingo

srun hostname
echo $CUDA_VISIBLE_DEVICES
echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7
echo $8
echo $9
echo ${10}
echo ${11}
echo ${12}
echo ${13}
echo ${14}
echo ${15}
echo ${16}

python -u whisper_decode_video.py --lang $1 \
                                --model-type $2 \
                                --noise-snr $3 \
                                --noise-fn $4 \
                                --checkpoint-path $5 \
                                --beam-size $6 \
                                --modalities $7 \
                                --use_av_hubert_encoder $8 \
                                --av_fusion $9 \
                                --fp16 ${10} \
                                --decode-path ${11} \
                                --av-hubert-path ${12} \
                                --av-hubert-ckpt ${13} \
                                --task ${14} \
                                --normalizer ${15} \
                                --use-original-whisper ${16}