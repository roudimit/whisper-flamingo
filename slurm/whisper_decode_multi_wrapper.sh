# Select model checkpoint
# Audio checkpoints
# checkpoint=whisper_multi-all_medium.pt  
# checkpoint=whisper_multi-all_small.pt  

# Audio-visual checkpoints
checkpoint=whisper-flamingo_multi-all_medium.pt
# checkpoint=whisper-flamingo_multi-all_small.pt

# Select model size
# model=large-v2
model=medium
# model=small

# Select modalities
# modalities=asr # whisper
modalities=avsr # whisper-flamingo

# Select av_fusion type
# av_fusion="None" # asr - use this for audio only whisper models
av_fusion="separate" # use this for whisper-flamingo models

# Select whether to use AV-HuBERT encoder
# use_av_hubert_encoder=0 # for whisper / asr
use_av_hubert_encoder=1 # for whisper-flamingo / avsr

# Select multilingual or EN babble noise
# noise_fn=noise/babble/muavic/test.tsv # multilingual babble noise 
noise_fn=noise/babble/lrs3/test.tsv # single lrs3 mixture
# noise_fn=/data/sls/scratch/roudi/datasets/musan/tsv/babble/test.tsv
# noise_fn=/data/sls/scratch/roudi/datasets/musan/tsv/music/test.tsv
# noise_fn=/data/sls/scratch/roudi/datasets/musan/tsv/noise/test.tsv 
# noise_fn=/data/sls/scratch/roudi/datasets/lrs3/noise/speech/test.tsv

# Select AV-HuBERT checkpoint
av_hubert_ckpt=models/mavhubert_only_weights.pt # multilingual
# av_hubert_ckpt=models/large_noise_pt_noise_ft_433h_only_weights.pt # english

# Specify Paths
checkpoint_root=models/
# checkpoint_root=models/checkpoint/
checkpoint_path=${checkpoint_root}${checkpoint}
decode_path=decode/
fp16=1
av_hubert_path=av_hubert/avhubert/

# for ASR only: ignore checkpoint path and use original whisper weights
use_original_whisper=0
# use_original_whisper=1

task=transcribe # ASR
normalizer=fairseq

for lang in en ar de el es fr it pt ru; do # all langs
# for lang in en es fr it pt; do # high resource langs
# for lang in ar de el ru; do # low resource langs
# for lang in en; do
# for beam_size in 1; do
for beam_size in 5; do
    for noise_snr in 0 1000; do
    # for noise_snr in 1000; do
    # for noise_snr in -10 -5 0 5 10; do
            echo $modalities $lang $noise_snr
            sbatch slurm/whisper_decode.sh $lang \
                                    $model \
                                    $noise_snr \
                                    $noise_fn \
                                    $checkpoint_path \
                                    $beam_size \
                                    $modalities \
                                    $use_av_hubert_encoder \
                                    $av_fusion \
                                    $fp16 \
                                    $decode_path \
                                    $av_hubert_path \
                                    $av_hubert_ckpt \
                                    $task \
                                    $normalizer \
                                    $use_original_whisper
        done
    done
done