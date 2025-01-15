# Select model checkpoint
# Audio checkpoints
# checkpoint=whisper_en-x_large/last.ckpt
# checkpoint=whisper_en-x_large.pt
# checkpoint=whisper_en_large.pt
# checkpoint=whisper_en-x_medium.pt
# checkpoint=whisper_en-x_small.pt

# Audio-visual checkpoints
# checkpoint=whisper_flamingo_en-x_large/last.ckpt
# checkpoint=whisper-flamingo_en-x_large.pt
# checkpoint=whisper-flamingo_en_large.pt
# checkpoint=whisper-flamingo_en-x_medium.pt
checkpoint=whisper-flamingo_en-x_small.pt

# Select model size
# model=large-v2
# model=medium
model=small

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
noise_fn=noise/babble/muavic/test.tsv # multilingual babble noise
# noise_fn=noise/babble/lrs3/test.tsv # single lrs3 mixture

# Select the task
task=En-X # En ASR and En-X translation
# task=transcribe # ASR only

# Select beam size
# beam_size=1
beam_size=15

# Fp16 1 for on 0 for off
fp16=1
# fp16=0

# Specify Paths
checkpoint_root=models/
# checkpoint_root=models/checkpoint/
checkpoint_path=${checkpoint_root}${checkpoint}
decode_path=decode/
av_hubert_path=av_hubert/avhubert/
av_hubert_ckpt=models/large_noise_pt_noise_ft_433h_only_weights.pt
use_original_whisper=0
normalizer=fairseq

for lang in en; do
# for lang in en ru el es fr it pt; do 
    for noise_snr in 1000 0; do
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