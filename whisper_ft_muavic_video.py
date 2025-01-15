import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
import pandas as pd
import whisper
import argparse
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    load_data,
    load_wave,
    load_video_feats,
    add_noise,
    WhisperVideoCollatorWithPadding,
    whisper_optimizer,
    whisper_video_projection_optimizer,
    whisper_flamingo_projection_optimizer,
    setup_logging_and_checkpoint,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import LengthBatchSampler

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class MuavicVideoDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, model_name, max_length, 
                 spec_augment, noise_prob=0, noise_fn=None, train=False, noise_snr=0) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.train = train
        self.noise_snr = noise_snr
        print("Dataloader max length : {}".format(max_length))
        print("Loaded {} noise wavs".format(len(self.noise_fn)))

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        lang, audio_path, text, _ = self.audio_info_list[id]
        # audio = load_wave(audio_path, sample_rate=self.sample_rate)
        if np.random.rand() > self.noise_prob: # disable noise
            sample_rate, wav_data = wavfile.read(audio_path)
            audio = wav_data.flatten().astype(np.float32) / 32768.0
        else: # add noise
            SNR = self.noise_snr            
            sample_rate, wav_data = wavfile.read(audio_path)
            audio = add_noise(wav_data, self.noise_fn, noise_snr=SNR).flatten().astype(np.float32) / 32768.0

        audio_frames = len(audio.flatten()) // 160
        # pad audio to cfg.audio_max_length (longer samples filtered out already)
        if self.max_length != None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels) # freq by time
    
        if self.spec_augment:
            if self.spec_augment == "ls-double":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames)).T # expects time by freq
            elif self.spec_augment == "ls-basic":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames, n_freq_mask=1, n_time_mask=1)).T # expects time by freq
            else:
                raise NotImplementedError 

        # Seems like Whisper decode always predicts first token with space, so add a space in the beginning
        # dec_input_ids = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(" " + text)
        dec_input_ids = [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format(lang)], 
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        video_path = audio_path.replace('audio', 'video').replace('.wav', '.mp4')
        video = load_video_feats(video_path, train=self.train)
        video = video.astype(np.float32)

        # Trim some videos longer than the audio
        max_video_len = round(len(audio.flatten()) / 16000 * 25)
        if len(video) > max_video_len:
            video = video[:max_video_len]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "video": video
        }

class WhisperVideoModule(LightningModule):
    def __init__(self, cfg, model_name, lang, train_dataset, val_dataset, test_dataset) -> None:
        super().__init__()
        self.model_name = model_name
        print("Loading Whisper model and weights")
        self.model = whisper.load_model(model_name,
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/data/sls/scratch/roudi/experiments/whisper/',
                                        dropout_rate=cfg.dropout_rate,
                                        video=True,
                                        video_model_path=cfg.video_model_ckpt, 
                                        prob_av=cfg.prob_use_av, 
                                        prob_a=cfg.prob_use_a,
                                        av_hubert_encoder=cfg.use_av_hubert_encoder,
                                        av_fusion=cfg.av_fusion,
                                        add_gated_x_attn=cfg.add_gated_x_attn,)
        if cfg.pt_ckpt != '': # load audio-only FT ckpt
            state_dict = torch.load(cfg.pt_ckpt, map_location=torch.device('cpu'))
            state_dict = state_dict['state_dict']
            state_dict_updated = {k[6:]: v  for k, v in state_dict.items()} # remove 'model.'
            print(state_dict_updated.keys())
            try:
                self.model.load_state_dict(state_dict_updated) 
            except BaseException as e: 
                print(str(e))
                print("Loading weights with strict=False")
                self.model.load_state_dict(state_dict_updated, strict=False) 
        self.freeze_video_model = cfg.freeze_video_model
        self.freeze_video_batch_norm_stats = cfg. freeze_video_batch_norm_stats
        multilingual = True if 'large' in model_name or 'en' not in model_name else False
        print("Multilingual tokenizer : {}".format(multilingual))
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=multilingual, task='transcribe')

        # if 'large' in self.model_name: # only decoder training
        #     for p in self.model.encoder.parameters():
        #         p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.train_dataset = train_dataset
        self.__val_dataset = val_dataset
        self.__test_dataset = test_dataset
        self.special_token_set = set(self.tokenizer.special_tokens.values())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        video = batch["video"]
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        padding_mask = batch["padding_mask"]

        if self.freeze_video_model: # freeze video encoder
            for param in self.model.encoder.video_model.parameters():
                param.requires_grad = False
        if self.freeze_video_batch_norm_stats: # use batch stats from ckpt (do not estimate on batch)
            self.model.encoder.video_model.eval()

        if self.cfg.add_gated_x_attn != 0: # freeze whisper encoder gradients for x-attn
            video_projection_layers = ["video_projection"] if self.cfg.freeze_video_model else ["video"]
            for n, p in self.model.encoder.named_parameters():
                if not any(nd in n for nd in video_projection_layers):
                    p.requires_grad = False
        # if 'large' in self.model_name: # only decoder training, NOTE: be careful with linear layer here
        #     with torch.no_grad():
        #         features, x_v = self.model.encoder(input_ids, video, training=True, padding_mask=padding_mask)
        # else:
        features, x_v = self.model.encoder(input_ids, video, training=True, padding_mask=padding_mask)

        out = self.model.decoder(dec_input_ids, features, xv=x_v)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
            
    def validation_step(self, batch, batch_id, dataloader_idx=None):
        video = batch["video"]
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        padding_mask = batch["padding_mask"]

        features_av, x_norm, x_v_norm_pre, x_v_norm_post, x_v = self.model.encoder(input_ids, video, track_norm=True,
                                                                              padding_mask=padding_mask)
        out_av = self.model.decoder(dec_input_ids, features_av, xv=x_v)

        if cfg.add_gated_x_attn == 0:
            features_a, x_v = self.model.encoder(input_ids, video, test_a=True, padding_mask=padding_mask)
            out_a = self.model.decoder(dec_input_ids, features_a)

            features_v, x_v = self.model.encoder(input_ids, video, test_v=True, padding_mask=padding_mask)
            out_v = self.model.decoder(dec_input_ids, features_v)

        labels[labels == -100] = self.tokenizer.eot

        mod_list = {"av": out_av, "a": out_a, "v": out_v} if cfg.add_gated_x_attn == 0 else {"av": out_av}
        for mod, out in mod_list.items():
            loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
            # remove all decoder predictions after first eot for proper decoding
            tokens = torch.argmax(out, dim=2)

            # Set all decoder predictions after first eot to eot
            # TODO: fix for large-v3, which predicts <eot> in the beginning
            eot_find = (torch.where(tokens == self.tokenizer.eot, 1, 0))
            first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1).cuda() * eot_find, dim=1, keepdim=True)
            tokens[torch.arange(eot_find.shape[1]).cuda() > first_eot] = self.tokenizer.eot

            # calculate next token prediction, not include lang tag, task, and no timestamps token
            mask = ~(tokens[:, 3:] == self.tokenizer.eot) # torch.ne fails for some reason
            n_correct = torch.sum(
                tokens[:, 3:].masked_select(mask).eq(labels[:, 3:].masked_select(mask))
            )
            total = torch.sum(mask)
            acc = n_correct.item()  / (total.item() + 1e-6)
            acc = acc if acc < 1 else 0

            o_list, o_list_full, l_list, l_list_full = [], [], [], []
            for o, l in zip(tokens, labels):
                o_list.append(self.tokenizer.decode([t for t in o if t.item() not in self.special_token_set]))
                # o_list_full.append(self.tokenizer.decode(o))
                l_list.append(self.tokenizer.decode([t for t in l if t.item() not in self.special_token_set]))
                # l_list_full.append(self.tokenizer.decode(l))
            wer, cer = wer_cer(hypo=o_list, ref=l_list)
        
            # for i, (hypo, hypo_full, ref, ref_full) in enumerate(zip(o_list, o_list_full, l_list, l_list_full)):
            print("Mod: {}".format(mod))
            for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
                print("-"*10)
                print("PRED: {}".format(hypo))
                # print(hypo_full)
                print("REF:  {}".format(ref))
                # print(ref_full)
                if i == 1: break

            # log_prefix = 'val' if dataloader_idx == 1 else 'val_noisy_en_babble'
            log_prefix = {0: 'test_noisy_en_babble', 1: 'test', 2: 'val_noisy_en_babble', 3: 'val'}
            self.log("{}/loss_{}".format(log_prefix[dataloader_idx], mod), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/cer_{}".format(log_prefix[dataloader_idx], mod), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/wer_{}".format(log_prefix[dataloader_idx], mod), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("{}/acc_{}".format(log_prefix[dataloader_idx], mod), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            # if self.cfg.add_gated_x_attn != 0 and dataloader_idx == 1: # only log for clean
            #     for i in range(0, 24, 6):
            #         self.log("val/attn_gate_layer_{}".format(i), self.model.decoder.blocks[i].attn_gate.tanh().item(), on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            #         self.log("val/ff_gate_layer_{}".format(i), self.model.decoder.blocks[i].ff_gate.tanh().item(), on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)         
        if dataloader_idx == 3: # only log for val,clean
            self.log("val/x_norm", x_norm, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("val/x_v_norm_pre", x_v_norm_pre, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
            self.log("val/x_v_norm_post", x_v_norm_post, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        
        return
    
    def configure_optimizers(self):
        model = self.model
        if self.cfg.add_gated_x_attn != 0:
            optimizer, scheduler = whisper_flamingo_projection_optimizer(model, self.cfg, self.t_total)
        elif self.cfg.video_projection_train_only:
            optimizer, scheduler = whisper_video_projection_optimizer(model, self.cfg, self.t_total)
        else:
            optimizer, scheduler = whisper_optimizer(model, self.cfg, self.t_total)
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        dataset = MuavicVideoDataset(self.__train_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=self.cfg.spec_augment,
                                      noise_prob=cfg.noise_prob,
                                      noise_fn=cfg.noise_fn,
                                      train=True,
                                      noise_snr=cfg.noise_snr_train,)  
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * self.cfg.batch_size),
                            shapes=[i[3] for i in self.__train_dataset],
                            sort_in_batch='descending',
                            sort_batch='shuffle',
                            drop_last=True,)
        if cfg.num_devices > 1:
            print("Using distributed sampler")
            length_sorter = DistributedSamplerWrapper(length_sorter)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperVideoCollatorWithPadding())

    def val_dataloader_clean(self):
        dataset = MuavicVideoDataset(self.__val_dataset, 
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=None,
                                spec_augment=False,
                                noise_prob=0,
                                train=False,)
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * 16),
                            shapes=[i[3] for i in self.__val_dataset],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperVideoCollatorWithPadding())

    def val_dataloader_noisy(self):
        dataset = MuavicVideoDataset(self.__val_dataset, 
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=None,
                                spec_augment=False,
                                noise_prob=1,
                                noise_fn=cfg.noise_fn_val,
                                train=False,)
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * 16),
                            shapes=[i[3] for i in self.__val_dataset],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperVideoCollatorWithPadding())
    
    def test_dataloader_clean(self):
        dataset = MuavicVideoDataset(self.__test_dataset, 
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=None,
                                spec_augment=False,
                                noise_prob=0,
                                train=False,)
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * 16),
                            shapes=[i[3] for i in self.__test_dataset],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperVideoCollatorWithPadding())

    def test_dataloader_noisy(self):
        dataset = MuavicVideoDataset(self.__test_dataset, 
                                self.tokenizer, 
                                SAMPLE_RATE,
                                self.model_name,
                                max_length=None,
                                spec_augment=False,
                                noise_prob=1,
                                noise_fn=cfg.noise_fn_test,
                                train=False,)
        length_sorter = LengthBatchSampler(batch_bins=int(self.cfg.audio_max_length * 16),
                            shapes=[i[3] for i in self.__test_dataset],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperVideoCollatorWithPadding())

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint(cfg.log_output_dir, 
                                                                                cfg.check_output_dir, 
                                                                                cfg.train_name, 
                                                                                cfg.train_id,
                                                                                cfg.monitor,)
    if cfg.lang == 'multi-all':
        audio_transcript_pair_list = load_data(cfg.audio_max_length, cfg.text_max_length, 
                                            ['en', 'ar', 'de', 'el', 'es', 'fr', 'it', 'pt', 'ru'],
                                            reduce_val=300, include_audio_lens=True)
    elif cfg.lang == 'multi':
        audio_transcript_pair_list = load_data(cfg.audio_max_length, cfg.text_max_length, 
                                            ['en', 'es', 'fr', 'it', 'pt'],
                                            reduce_val=300, include_audio_lens=True)
    elif cfg.lang == 'multi_en-st':
        audio_transcript_pair_list = load_data(cfg.audio_max_length, cfg.text_max_length, 
                                           ['en', 'el', 'es', 'fr', 'it', 'pt', 'ru'],
                                           reduce_val=200, include_audio_lens=True, task='En-X')
    elif 'lrs2' in cfg.lang:
        audio_transcript_pair_list = load_data(cfg.audio_max_length, cfg.text_max_length, ['en'], 
                                            include_audio_lens=True, lrs2=True)
    else:
        audio_transcript_pair_list = load_data(cfg.audio_max_length, cfg.text_max_length, 
                                               [cfg.lang], include_audio_lens=True, vc2=cfg.vc2, vc2_path=cfg.vc2_path)

    model = WhisperVideoModule(cfg, cfg.model_name, cfg.lang, 
                               audio_transcript_pair_list['train'], 
                               audio_transcript_pair_list['valid'],
                               audio_transcript_pair_list['test'])
    
    strategy = DDPStrategy(find_unused_parameters=True) if cfg.num_devices > 1 else "auto"
    trainer = Trainer(
        precision=16,
        strategy=strategy,
        accelerator="gpu",
        max_steps=cfg.num_train_steps,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
        num_sanity_val_steps=0, # default is 2 batches, 0 to turn off
        devices=cfg.num_devices,
        val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps), # validate after this number batches
        # val_check_interval=cfg.validate_every_n_batches, # validate after this number batches
        check_val_every_n_epoch=None, # If None, validation will be done solely based on the number of training batches
        reload_dataloaders_every_n_epochs=1, # shuffle the dataloader after an epoch
        # gradient_clip_val=1, # TODO: add as config variable?
        use_distributed_sampler=False, # implemented custom distributed trainer
        sync_batchnorm=True,
    )

    # TODO: save config file tp the checkpoint dir, also for pre-trained model
    print(cfg)
    resume_ckpt = f"{cfg.check_output_dir}/{cfg.train_id}/last.ckpt"
    if os.path.exists(resume_ckpt) and cfg.resume_training: # resume training, don't validate
        trainer.fit(model, ckpt_path='last', val_dataloaders=[model.test_dataloader_noisy(), model.test_dataloader_clean(),
                                                   model.val_dataloader_noisy(), model.val_dataloader_clean()])
    else:
        trainer.validate(model=model, dataloaders=[model.test_dataloader_noisy(), model.test_dataloader_clean(),
                                                   model.val_dataloader_noisy(), model.val_dataloader_clean()]) # validate before training
        trainer.fit(model, val_dataloaders=[model.test_dataloader_noisy(), model.test_dataloader_clean(),
                                                   model.val_dataloader_noisy(), model.val_dataloader_clean()])

