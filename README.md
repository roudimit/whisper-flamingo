# Whisper-Flamingo

## Updates
**Nov 19, 2024:** We achieved SOTA ASR (1.3\%) and SOTA AVSR (1.4\%) on LRS2 - checkpoints are released below.  
**Oct 11, 2024:** We achieved SOTA ASR (0.68% WER) and SOTA AVSR (0.72% WER) on LRS3 by training on LRS3 and VoxCeleb2 - checkpoints are released below.

## Introduction

Integrating Visual Features into Whisper for Audio-Visual Speech Recognition and Translation

We propose Whisper-Flamingo which integrates visual features into the Whisper speech recognition and translation model with gated cross attention. 
Our audio-visual Whisper-Flamingo outperforms audio-only Whisper on English speech recognition and En-X translation for 6 languages in noisy conditions.
Moreover, Whisper-Flamingo is a versatile model and conducts all of these tasks using one set of parameters, while prior methods are trained separately on each language.

![Whisper-Flamingo](assets/whisper_flamingo_fig.jpg "Whisper-Flamingo")

# Video Demos
Check out the video demo below (turn sound on).
We made several videos about Whisper-Flamingo:
- 30s demo of Whisper-Flamingo (same video below): [YouTube link](https://youtu.be/EsFlaqYVkro)
- 2m demo comparing Whisper and Whisper-Flamingo: [YouTube link](https://youtu.be/elHF-EQgmNs)
- 10m presentation: [YouTube link](https://youtu.be/MemXz2IqwIM)

<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/roudimit/whisper-flamingo/assets/16767254/7ce5b2c3-4d21-4453-8bd2-8c4977c948f9" muted="false"></video>
    </td>
</tr>
</table>

# Colab Demos
We support two colab demos (local copies in `./notebooks`):
- Test Whisper-Flamingo on an example audio / video [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rnhNOZuUxh-WXXloo_z1fu5DKeJrH95p)
- Reproduce our results on LRS3 / MuAViC: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tYI_7GxJuQdhWnO4m6TUplEoxVaYgbvW)

# Virtual Environment for Training and Testing
Since this project uses the MuAViC dataset, we base our virtual environment on theirs.

Create a fresh virtual environment:
```
conda create -n whisper-flamingo python=3.8 -y
conda activate whisper-flamingo
```
Clone MuAViC repo and install their requirements: 
```
conda install -c conda-forge ffmpeg==4.2.2 -y
conda install -c conda-forge sox -y
git clone https://github.com/facebookresearch/muavic.git muavic-setup
cd muavic-setup
pip install -r requirements.txt
cd ..
```
Clone the "muavic" branch of av_hubert's repo and install Fairseq:
```
git clone -b muavic https://github.com/facebookresearch/av_hubert.git
cd av_hubert
git submodule init
git submodule update
# Install av-hubert's requirements
pip install -r requirements.txt
# Install fairseq
cd fairseq
pip install --editable ./
cd ../..
```
Install extra packages used in our project:
```
pip install tiktoken==0.5.2 pytorch-lightning==2.1.3 numba==0.58.1 transformers==4.36.2 evaluate tensorboardX
```

# Download and prepare data
LRS3 / MuAViC: We provide all data to reproduce the results on the test set. For instructions on how to prepare the LRS3 training set (and more details about the test noise), see `preparation/README.md`. 

Download and extract our resources:
```
wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/muavic.tar.gz
wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/noise.tar.gz
tar -xf muavic.tar.gz
tar -xf noise.tar.gz
echo $(pwd)/noise/babble/muavic/babble_all.wav > ./noise/babble/muavic/test.tsv
echo $(pwd)/noise/babble/muavic/babble_all.wav > ./noise/babble/muavic/valid.tsv
echo $(pwd)/noise/babble/lrs3/noise.wav > ./noise/babble/lrs3/test.tsv
echo $(pwd)/noise/babble/lrs3/noise.wav > ./noise/babble/lrs3/valid.tsv
```

LRS2: The data can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) after signing a license and sending it to the BBC (helper script: `notebooks/lrs2_download.ipynb`). In our experience, it took a week to receive the username & password for the data download. We used the [AutoAVSR scripts](https://github.com/mpc001/auto_avsr/tree/main/preparation) to process LRS2 (using the provided facial landmarks). Finally, the AutoAVSR data lists must be converted to AV-HuBERT / Fairseq manifests. We provide a script to do this (`notebooks/lrs2_make_tsv.ipynb`).

# Pre-trained Models
We release our pre-trained models (GPUs = GPUs used for training).
- Our audio models are fine-tuned with noise from MUSAN and LRS3 (including babble noise, speech, and music), making them perform better in noise (see the paper and our video demo for more details)
- We also release the models trained on the combination of LRS3 and VoxCeleb2 (the transcripts of VoxCeleb2 were obtained by Whisper Large-V2, available from this [repo](https://github.com/nikvaessen/disjoint-mtl?tab=readme-ov-file)). We release the models fine-tuned with noise (noisy) and without noise (clean). **whisper_en_large_vc2_clean achieves SOTA ASR on LRS3 (0.68% WER) and whisper-flamingo_en_large_vc2_clean achieves SOTA AVSR on LRS3 (0.72% WER).**
- LRS2 models: these models are trained on the LRS2 dataset with noise added from MUSAN and LRS3. **whisper_lrs2_medium achieves SOTA ASR on LRS2 (1.3% WER) and whisper-flamingo_lrs2_medium achieves SOTA AVSR on LRS2 (1.4%)**.
- Our LRS3 models support transcription in English (En) and En-X translation into 6 languages: Greek (El), Spanish (Es), French (Fr), Italian (It), Portuguese (Pt), and Russian (Ru).
Note that to enable the new En-X translation capabilities, we use the 'transcribe' token instead of the 'translate' token as input to the decoder since the latter was already used for X-En translation.
- For English, our models don't output punctuation and capitalization since the LRS3 English training text removed them. For En-X translation, our models output punctuation and capitalization since they were retained in the training translations.


### Audio-only Whisper (fine-tuned on LRS3 / MuAViC)
|   Mod.  |   Size  |   VoxCeleb2 | Parameters  |   En ASR  |   En-X ST  |   GPUs  |   Download Link  |
|---|---|---|---|---|---|---|---|
|   A  |   Large-V2  | yes |   1,550M  |   yes  |   no  |   1x A6000, 48GB  |   noisy: [whisper_en_large_vc2_noisy](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_en_large_vc2_noisy.pt) <br> clean: [whisper_en_large_vc2_clean](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_en_large_vc2_clean.pt) |
|   A  |   Large-V2  | no |   1,550M  |   yes  |   no  |   1x A6000, 48GB  |   [whisper_en_large](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_en_large.pt)  |
|   A  |   Large-V2  | no |   1,550M  |   yes  |   yes  |   4x A6000, 48GB  |   [whisper_en-x_large](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_en-x_large.pt)  |
|   A  |   LRS2-Medium  | no |   769M  |   yes  |   no  |   1x A6000, 48GB  |   [whisper_lrs2_medium](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_lrs2_medium.pt)  |
|   A  |   Medium | no |   769M  |   yes  |   yes  |   4x A5000, 24GB  |   [whisper_en-x_medium](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_en-x_medium.pt)  |
|   A  |   Small  | no |   244M  |   yes  |   yes  |   4x A5000, 24GB  |   [whisper_en-x_small](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_en-x_small.pt)  |

### Audio-visual Whisper-Flamingo
|   Mod.  |   Size  |   VoxCeleb2 | Parameters  |   En ASR  |   En-X ST  |   GPUs  |   Download Link  |
|---|---|---|---|---|---|---|---|
|   AV  |   Large-V2  | yes |    2,497M  |   yes  |   no  |   1x A6000, 48GB  |   noisy: [whisper-flamingo_en_large_vc2_noisy](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en_large_vc2_noisy.pt) <br> clean: [whisper-flamingo_en_large_vc2_clean](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en_large_vc2_clean.pt)   |
|   AV  |   Large-V2  | no |    2,497M  |   yes  |   no  |   1x A6000, 48GB  |   [whisper-flamingo_en_large](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en_large.pt)  |
|   AV  |   Large-V2  | no |    2,497M  |   yes  |   yes  |   4x A6000, 48GB  |   [whisper-flamingo_en-x_large](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en-x_large.pt)  |
|   AV  |   LRS2-Medium  | no |   1,390M  |   yes  |   no  |   1x A6000, 48GB  |   [whisper-flamingo_lrs2_medium](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_lrs2_medium.pt)  |
|   AV  |   Medium  | no |   1,390M  |   yes  |   yes  |   4x A6000, 48GB  |   [whisper-flamingo_en-x_medium](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en-x_medium.pt)  |
|   AV  |   Small  | no |    651M  |   yes  |   yes  |   4x A5000, 24GB  |   [whisper-flamingo_en-x_small](https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en-x_small.pt)  |

# Decoding Script

The script `whisper_decode_video.py` is used for decoding both audio-only Whisper models and audio-visual Whisper-Flamingo models. We also provide a SLURM scripts to run decoding in parallel, see the next section for details.
### Audio-Only Decoding
Download our audio-only Whisper model fine-tuned for En-X translation.
```
mkdir models
wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_en-x_small.pt -P models
```

Decode an audio-only model (see `whisper_decode_video.py` for argument details):
- To switch to En-X translation, change the `lang` to the target language.
- Here we use babble noise from MuAViC at 0 SNR. Use `noise/babble/lrs3/test.tsv` for babble noise from LRS3. Use `--noise-snr 1000` to evaluate in clean conditions.
- Here use beam size 1. In the paper we report results with beam size 15.
- For GPU without fp16, and for cpu, use `--fp16 0`
```
python -u whisper_decode_video.py --lang en \
                                --model-type small \
                                --noise-snr 0 \
                                --noise-fn noise/babble/muavic/test.tsv \
                                --beam-size 1 \
                                --modalities asr \
                                --fp16 1 \
                                --checkpoint-path models/whisper_en-x_small.pt \
                                --decode-path decode/
```

LRS2 ASR decoding (adjust `noise_snr` as desired):
```
python -u whisper_decode_video.py --lang lrs2 \
                                --model-type medium \
                                --noise-snr 1000 \
                                --noise-fn noise/babble/lrs3/test.tsv \
                                --beam-size 1 \
                                --modalities asr \
                                --fp16 1 \
                                --checkpoint-path models/whisper_lrs2_medium.pt \
                                --decode-path decode/
```

### Audio-Visual Decoding
Download our audio-visual Whisper-Flamingo model fine-tuned for En-X translation.
Note: the AV-HuBERT weights must be downloaded and are used by Fairseq to load the architecture.
```
mkdir models
wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en-x_small.pt -P models
wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/large_noise_pt_noise_ft_433h_only_weights.pt -P models
```

Decode an audio-visual model:
```
python -u whisper_decode_video.py --lang en \
                                --model-type small \
                                --noise-snr 0 \
                                --noise-fn noise/babble/muavic/test.tsv \
                                --beam-size 1 \
                                --modalities avsr \
                                --use_av_hubert_encoder 1 \
                                --av_fusion separate \
                                --fp16 1 \
                                --checkpoint-path models/whisper-flamingo_en-x_small.pt \
                                --decode-path decode/ \
                                --av-hubert-path av_hubert/avhubert/ \
                                --av-hubert-ckpt models/large_noise_pt_noise_ft_433h_only_weights.pt
```

LRS2 AVSR decoding (adjust `noise_snr` as desired):
```
python -u whisper_decode_video.py --lang lrs2 \
                                --model-type medium \
                                --noise-snr 1000 \
                                --noise-fn noise/babble/lrs3/test.tsv \
                                --beam-size 1 \
                                --modalities avsr \
                                --use_av_hubert_encoder 1 \
                                --av_fusion separate \
                                --fp16 1 \
                                --checkpoint-path models/whisper-flamingo_lrs2_medium.pt \
                                --decode-path decode/ \
                                --av-hubert-path av_hubert/avhubert/ \
                                --av-hubert-ckpt models/large_noise_pt_noise_ft_433h_only_weights.pt
```

# Decoding Script in Parallel with SLURM
We provide `slurm/whisper_decode_video_slurm_wrapper.sh` which submits decoding jobs tp SLURM for a given checkpoint to test all En-X languages in both clean / noisy conditions. Please modify `slurm/whisper_decode_video_slurm.sh` to match your SLURM environment.

After submitting all jobs with `source slurm/whisper_decode_video_slurm_wrapper.sh`, use `slurm/check_results.ipynb` to print the results of all decoding runs. It will load the decoding WER / BLEU scores and print them in a convinient table.

# Training

### Step 1: Fine-tune audio-only Whisper for En-X translation on MuAViC
First, in `config/audio/audio_en-x_large.yaml`, replace `noise_fn: '/data/sls/scratch/roudi/datasets/musan/tsv/all/train.tsv'` with the path to your training noise.
Command:
```
python -u whisper_ft_muavic.py config/audio/audio_en-x_large.yaml
```
We also provide a slurm script in `slurm/train_audio_4gpu.sh`. 
It took about 2-3 days to fine-tune Whisper Large-V2 on our GPUs. 
The medium and small models are faster take less time to train.

### Step 2: Train audio-visual Whisper-Flamingo with gated cross attention
Once the audio model is fine-tuned, we freeze the weights and insert the gated cross-attention layers to train the audio-visual Whisper-Flamingo. Command:
```
python -u whisper_ft_muavic_video.py config/audio-visual/av_en-x_large.yaml
```
We also provide a slurm script in `slurm/train_video_4gpu.sh`.
Training Whisper-Flamingo is faster since the cross-attention layers are the only trainable layers. It took about 1 day to train Whisper-Flamingo Large on our GPUs (not including the time to fine-tune the audio model in the first step.).

### Training progress
Model weights will be saved in `models/checkpoint`.
Tensorboard can be opened to monitor several metrics.
```
cd slurm
tensorboard --logdir .  --port 6008
```
### Training notes
- Training should work on 1 GPU or multiple GPUs, although some settings need to be adjusted (such as batch size)
- The original Whisper code always pads audio to 30s. We avoid this and instead batch together samples of similar length and pad to the longest sample in the batch (this minimizes padding).


# Acknowledgments
This code based is based on the following repos: [Whisper Fine-Tuning Demo](https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz?usp=sharing), [Whisper](https://github.com/openai/whisper), [AV-HuBERT](https://github.com/facebookresearch/av_hubert), [MuAViC](https://github.com/facebookresearch/muavic), [ESPnet](https://github.com/espnet/espnet), [AutoAVSR](https://github.com/mpc001/auto_avsr), [Flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch).


# License
Our work is licensed under BSD-3. However, please check the licenses of the works we build on, including AV-HuBERT.

# Citation
```bibtex
@inproceedings{rouditchenko24_interspeech,
  title     = {Whisper-Flamingo: Integrating Visual Features into Whisper for Audio-Visual Speech Recognition and Translation},
  author    = {Andrew Rouditchenko and Yuan Gong and Samuel Thomas and Leonid Karlinsky and Hilde Kuehne and Rogerio Feris and James Glass},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {2420--2424},
  doi       = {10.21437/Interspeech.2024-322},
  issn      = {2958-1796},
}
```
