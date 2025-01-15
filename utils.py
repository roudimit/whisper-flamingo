import os
import cv2
import random
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as at
import numpy as np
import editdistance
from scipy.io import wavfile
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from operator import itemgetter
from typing import Iterator, Optional
from torch.utils.data import Dataset, DistributedSampler
from torch.utils.data.sampler import Sampler

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

def load_video_feats(video_path, train=False, image_crop_size=88, 
               image_mean=0.421, image_std=0.165):
    feats = load_video_av_hubert(video_path)
    if train:
        transform = Compose([
            Normalize( 0.0,255.0 ),
            RandomCrop((image_crop_size, image_crop_size)),
            HorizontalFlip(0.5),
            Normalize(image_mean, image_std)])
    else:
        transform = Compose([
            Normalize( 0.0,255.0 ),
            CenterCrop((image_crop_size, image_crop_size)),
            Normalize(image_mean, image_std)])
    feats = transform(feats)
    feats = np.expand_dims(feats, axis=-1) # T, H, W, C
    return feats

def load_video_av_hubert(path):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)

class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames

def select_noise(noise_wavs):
    rand_indexes = np.random.randint(0, len(noise_wavs), size=1)
    noise_wav = []
    for x in rand_indexes:
        noise_wav.append(wavfile.read(noise_wavs[x])[1].astype(np.float32))
    return noise_wav[0]

def add_noise(clean_wav, noise_wavs, noise_snr=0):
    clean_wav = clean_wav.astype(np.float32)
    noise_wav = select_noise(noise_wavs)
    if type(noise_snr) == int or type(noise_snr) == float:
        snr = noise_snr
    elif type(noise_snr) == tuple:
        snr = np.random.randint(noise_snr[0], noise_snr[1]+1)
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)]
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10**(snr/20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav

    #Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
            reduction_rate = max_int16 / mixed.max(axis=0)
        else :
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    mixed = mixed.astype(np.int16)
    return mixed

def load_data(AUDIO_MAX_LENGTH, TEXT_MAX_LENGTH, langs=['en', 'ar', 'de', 'el', 'es', 'fr', 'it', 'pt', 'ru'],
              muavic_root='/data/sls/scratch/roudi/datasets/muavic/', reduce_val=None, include_audio_lens=False,
              AUDIO_MAX_LENGTH_VAL=480000, vc2=False, vc2_path='', lrs2=False, visible=False, task='transcribe'):
    # reduce_val: If not None, keep this number of samples from the validation set
    audio_transcript_pair_list = {'train':[], 'valid':[], 'test':[]}
    for lang in langs:
        for split in audio_transcript_pair_list:
            if lrs2:
                txt_fn = os.path.join('/data/sls/scratch/roudi/datasets/lrs2/whisper-flamingo/{}.wrd'.format(split))
                tsv_fn = os.path.join('/data/sls/scratch/roudi/datasets/lrs2/whisper-flamingo/{}.tsv'.format(split))
            elif lang == 'en':
                if split == 'train' and vc2:
                    txt_fn = os.path.join(muavic_root, 'muavic', vc2_path, '{}.{}'.format(split, lang))
                    tsv_fn = os.path.join(muavic_root, 'muavic', vc2_path, '{}.tsv'.format(split))
                else:
                    tsv_fn = os.path.join(muavic_root, 'muavic', 'en', '{}.tsv'.format(split))
                    txt_fn = os.path.join(muavic_root, 'muavic', 'en', '{}.en'.format(split))
            else: # multilingual
                if task == 'transcribe':
                    if split == 'train' and vc2:
                        tsv_fn = os.path.join(muavic_root, 'muavic', lang, 'muavic_normalized', 'train_muavic_vc2.tsv')
                        txt_fn = os.path.join(muavic_root, 'muavic', lang, 'muavic_normalized', 'train_muavic_vc2.{}'.format(lang))
                    else:
                        tsv_fn = os.path.join(muavic_root, 'muavic', lang, 'muavic_normalized', '{}.tsv'.format(split))
                        txt_fn = os.path.join(muavic_root, 'muavic', lang, 'muavic_normalized', '{}.{}'.format(split, lang))
                elif task == 'En-X': # EN-X translation                    
                    tsv_fn = os.path.join(muavic_root, 'muavic', 'en', lang, '{}.tsv'.format(split))
                    txt_fn = os.path.join(muavic_root, 'muavic', 'en', lang, '{}.{}'.format(split, lang))
                elif task == 'X-En': # X-En translation
                    tsv_fn = os.path.join(muavic_root, 'muavic', lang, 'en', '{}.tsv'.format(split))
                    txt_fn = os.path.join(muavic_root, 'muavic', lang, 'en', '{}.en'.format(split))
                
            with open(tsv_fn) as tsv:
                with open(txt_fn) as txt:
                    audio_lns = tsv.readlines()[1:]
                    txt_lns = txt.readlines()
                    # audio path, audio length, text, text length, video_length
                    wav_fns = [(audio.strip().split('\t')[2],  int(audio.strip().split('\t')[-1]), txt.strip(), 
                                len(txt.strip()), int(audio.strip().split('\t')[-2])) for audio, txt in zip(audio_lns, txt_lns)]
                    pre_video_check = len(wav_fns)
                    wav_fns =  list(filter(lambda x: x[4] > 0, wav_fns))
                    post_video_check = len(wav_fns)
                    print("Removed {} samples with missing video (before filtering lengths)".format(pre_video_check - post_video_check))
                    pre_video_check = len(wav_fns)
                    wav_fns =  list(filter(lambda x: x[3] > 0, wav_fns))
                    post_video_check = len(wav_fns)
                    print("Removed {} samples with missing text".format(pre_video_check - post_video_check))
                    len_before = len(wav_fns)
                    if split == 'train': 
                        wav_fns =  list(filter(lambda x: x[1] <= AUDIO_MAX_LENGTH, wav_fns))
                        wav_fns =  list(filter(lambda x: x[3] <= TEXT_MAX_LENGTH, wav_fns))
                    elif split == 'valid': # whisper pos. embedding only up to 30s long, don't filter test
                        wav_fns =  list(filter(lambda x: x[1] <= AUDIO_MAX_LENGTH_VAL, wav_fns))
                    print("Total hours {} : {}".format(split, sum([int(x[1]) for x in wav_fns]) / 16000 / 3600))
                    if not include_audio_lens:
                        lang_filtered = [(lang, i[0], i[2]) for i in wav_fns]
                    else: 
                        lang_filtered = [(lang, i[0], i[2], i[1]) for i in wav_fns]
                    if split == 'valid' or split == 'test' and reduce_val is not None:
                        lang_filtered = lang_filtered[:reduce_val]
                    len_after = len(lang_filtered)
                    audio_transcript_pair_list[split] += lang_filtered
            print(lang, split, len_before, len_after)
    print("Total data lengths")
    print(len(audio_transcript_pair_list['train']))
    print(len(audio_transcript_pair_list['valid']))
    print(len(audio_transcript_pair_list['test']))
    return audio_transcript_pair_list

class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len =  max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length) # seems redundant

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

        return batch
    
class WhisperVideoCollatorWithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, video = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            video.append(f["video"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len =  max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length) # seems redundant

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        # 0 pad the videos
        video_lengths = [len(vid) for vid in video]
        max_video_len = max(video_lengths)
        video = [np.pad(vid, ((0, max_video_len - vid_len), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=0) for vid, vid_len in zip(video, video_lengths)]
        padding_mask = create_padding_mask(max_video_len, [max_video_len - vid_len for vid_len in video_lengths])
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "video": video,
            "padding_mask": padding_mask,
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch['video'] = batch['video'].permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]

        return batch
    
def create_padding_mask(T, padding_amounts):
    """
    Creates a padding mask for a batch of B x T tensors, given padding amounts.

    Args:
        padding_amounts: A list or tensor of integers, where each element
                         specifies the amount of padding for the corresponding
                         sequence in the batch.

    Returns:
        A PyTorch tensor of shape (B, T) containing 1s for padded elements and 0s
        for non-padded elements.
    """

    padded_lens = T - torch.tensor(padding_amounts, dtype=torch.long)[:, None]  # Add a dimension for broadcasting
    mask = padded_lens <= torch.arange(T, dtype=torch.long)[None, :]  # Add a dimension for broadcasting
    return mask

def whisper_optimizer(model, cfg, t_total, video=True):
    no_decay = ["bias", "LayerNorm.weight"]
    projection = ["video_projection"] # linear layer and scalar
    if video and cfg.video_projection_separate_lr != '': # ft video projection separate lr
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in projection)],
                "lr": cfg.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in projection)],
                "lr": cfg.video_projection_separate_lr,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=cfg.learning_rate,
                        eps=cfg.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    return optimizer, scheduler

def whisper_video_projection_optimizer(model, cfg, t_total):
    if cfg.video_projection_linear_scale != 1.0:
        print("Scaling video projection scaler by {}".format(cfg.video_projection_linear_scale))
        print(model.encoder.video_projection_scalar)
        with torch.no_grad():
            model.encoder.video_projection_scalar *= cfg.video_projection_linear_scale
        print(model.encoder.video_projection_scalar)

    optimizer_grouped_parameters = [
        {
            "params": [*model.encoder.video_projection.parameters(),
                       model.encoder.video_projection_scalar],
            "lr" : cfg.video_projection_lr, 
            "weight_decay": cfg.weight_decay,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=cfg.learning_rate,
                        eps=cfg.adam_epsilon,
                        weight_decay=cfg.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    return optimizer, scheduler

def whisper_flamingo_projection_optimizer(model, cfg, t_total):
    video_projection = ["video_projection"]
    x_attn = ["gated_x_attn", "attn_gate", "ff"] if cfg.freeze_video_model else ["video_model", "gated_x_attn", "attn_gate", "ff"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in x_attn + video_projection)],
            "lr": cfg.learning_rate,
        },
    ]
    print("optimizing params: ")
    print([n for n, p in model.named_parameters()
                        if any(nd in n for nd in x_attn + video_projection)])
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=cfg.learning_rate,
                        eps=cfg.adam_epsilon,
                        weight_decay=cfg.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    return optimizer, scheduler

def setup_logging_and_checkpoint(log_output_dir, check_output_dir, train_name, train_id, monitor='val/acc'):
    Path(log_output_dir).mkdir(exist_ok=True)
    Path(check_output_dir).mkdir(exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename="step-{step:05d}-wer={val/wer:.4f}-acc={val/acc:.4f}",
        monitor=monitor,
        mode='max',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    monitor = monitor.replace('test', 'val') if 'test' in monitor else monitor.replace('val', 'test')
    val_checkpoint = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename="step-{step:05d}-wer={val/wer:.4f}-acc={val/acc:.4f}",
        monitor=monitor,
        mode='max',
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    latest_checkpoint = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename="step-{step:05d}-wer={val/wer:.4f}-acc={val/acc:.4f}",
        monitor="step",
        mode='max',
        every_n_train_steps=5000,
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    callback_list = [checkpoint_callback,
                     val_checkpoint,
                     latest_checkpoint, 
                     LearningRateMonitor(logging_interval="step")]
    # callback_list = [checkpoint_callback,
    #                  LearningRateMonitor(logging_interval="step")]
    return tflogger, checkpoint_callback, callback_list

def wer_cer(hypo, ref):
    c_err, c_len, w_err, w_len = 0, 0, 0, 0
    for h, r in zip(hypo, ref):
        pred_words = h.split()
        pred_units = h.replace(' ', '|').replace('', ' ').split() # chars-space separated
        
        gt_words = r.split()
        gt_units = r.replace(' ', '|').replace('', ' ').split() # chars-space separated\
        c_err += editdistance.eval(pred_units, gt_units)
        c_len += len(gt_units)

        w_err += editdistance.eval(pred_words, gt_words)
        w_len += len(gt_words)
    return w_err/w_len, c_err/c_len

# https://github.com/mpc001/auto_avsr/blob/main/datamodule/samplers.py
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()

        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.sampler.set_epoch(epoch)

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)