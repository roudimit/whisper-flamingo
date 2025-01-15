import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .resnet import ResEncoder
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, 
                 add_adapter: bool = False, adapter_dim: int = 256, add_gated_x_attn: int = 0):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        
        # https://github.com/lucidrains/flamingo-pytorch/blob/10913abbc8b2ceabb2320560d7d9b85fcb85eee3/flamingo_pytorch/flamingo_pytorch.py#L207
        self.add_gated_x_attn = add_gated_x_attn
        if self.add_gated_x_attn != 0:
            print("Adding gated x attn layers")
            self.gated_x_attn = MultiHeadAttention(n_state, n_head)
            self.gated_x_attn_ln = LayerNorm(n_state)
            self.attn_gate = nn.Parameter(torch.tensor([0.]))
            
            self.ff_ln = LayerNorm(n_state)
            self.ff = nn.Sequential(
                Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
            )
            self.ff_gate = nn.Parameter(torch.tensor([0.]))  
        
    def apply_gated_x_attn(self, x, xv):
        x = x + self.gated_x_attn(self.gated_x_attn_ln(x), xv)[0] * self.attn_gate.tanh()
        x = x + self.ff(self.ff_ln(x)) * self.ff_gate.tanh()
        return x

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        xv: Optional[Tensor] = None,
    ):
        if self.add_gated_x_attn != 0: 
            x = self.apply_gated_x_attn(x, xv)
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))        
        return x
    
class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, 
              dropout_rate: float, video: bool, video_model_path: str, av_hubert_path: str,
              prob_av: float, prob_a: float, av_hubert_encoder: bool, av_fusion: str,
              add_adapter: bool, adapter_dim: int,
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, False,
                                    add_adapter, adapter_dim, add_gated_x_attn=0) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)      
        self.video = video
        self.av_hubert_encoder = av_hubert_encoder
        self.av_fusion = av_fusion
        self.video_model_path = video_model_path
        if video:
            self.video_projection_scalar = nn.Parameter(torch.tensor(1.))
            self.prob_av, self.prob_a = prob_av, prob_a
            if not av_hubert_encoder:
                self.video_projection = Linear(512, n_state)
                self.video_model = ResEncoder('prelu', video_model_path)
            else:
                from fairseq import checkpoint_utils, utils
                from argparse import Namespace
                self.video_projection = Linear(1024, n_state) # assuming AV-HuBERT large model
                utils.import_user_module(Namespace(user_dir=av_hubert_path))
                print("Loading AV-HuBERT encoder")
                load_weights = False if "no_weights" in video_model_path else True
                models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([video_model_path],) 
                                                                                        # load_weights=load_weights)
                self.video_model = models[0].encoder if 'ft' in video_model_path else models[0]
                num_parameters = sum(p.numel() for p in self.video_model.parameters())
                print("Using AV-HuBERT encoder with parameters: {}".format(num_parameters)) 
            if self.av_fusion == "lip-reader":
                self.video_projection_blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                            [ResidualAttentionBlock(n_state, n_head) for _ in range(3)]
                    )
                num_parameters = sum(p.numel() for p in self.video_projection_blocks.parameters())
                print("Adding visual transformer layers with number of params: {}".format(num_parameters)) 

    def forward(self, x: Tensor, x_v=None, training=False, test_a=False, test_v=False, track_norm=False, 
                padding_mask=None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        if not test_v:
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)
            if track_norm:
                x_norm = torch.linalg.norm(x, dim=-1).mean()

        if self.video and not test_a:
            if not self.av_hubert_encoder:
                x_v = self.video_model(x_v) # B, F, T
                x_v = x_v.permute(0, 2, 1) # B, T, F
            elif not 'ft' in self.video_model_path: # AV-HuBERT ssl
                x_v = self.video_model(source={'video': x_v, 'audio': None}, 
                                        padding_mask=padding_mask, 
                                        mask=False, 
                                        features_only=True)
                x_v = x_v['x']
            else:
                x_v = self.video_model(source={'video': x_v, 'audio': None}, padding_mask=padding_mask)
                x_v = x_v['encoder_out'].permute(1, 0 , 2) # T, B, F -> B, T, F
                
            if track_norm:
                x_v_norm_pre = torch.linalg.norm(x_v, dim=-1).mean()

            if self.av_fusion == "lip-reader":
                x_v = torch.repeat_interleave(x_v, 2, dim=1) # 25 Hz -> 50 Hz 
            x_v = self.video_projection(x_v)
            x_v = self.video_projection_scalar * x_v

            if self.av_fusion == "lip-reader":
                # NOTE: pos embedding added before
                if x_v.shape[1] > 1500:
                    x_v = x_v[ :, :1500, :]
                # NOTE: if max_len is 30s, then the cropping doesn't do anything.
                x_v = (x_v + self.positional_embedding[: x_v.shape[1]]).to(x_v.dtype) # trim pos embedding

                for layer, block in enumerate(self.video_projection_blocks): # NOTE: new transformer layers
                    x_v = block(x_v)

                x = x_v # NOTE: use AV-HuBERT output as input

            if track_norm:
                x_v_norm_post = torch.linalg.norm(x_v, dim=-1).mean()

        if not test_v:
            # NOTE: pos embedding has max length of 1500 (30s after conv downsample from 3000 mel frames)
            if x.shape[1] > 1500:
                x = x[ :, :1500, :]

            # NOTE: if max_len is 30s, then the cropping doesn't do anything.
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype) # trim pos embedding

        for layer, block in enumerate(self.blocks):
            x = block(x)

        x = self.ln_post(x)

        if training: # modality dropout, encoder
            mod_drop_prob = np.random.random()
            if 0 < mod_drop_prob <= self.prob_av:
                pass # use both modalities
            elif self.prob_av < mod_drop_prob <= self.prob_av + self.prob_a:
                x_v = 0 * x_v # drop video
            else:
                x = 0 * x # drop audio
        if track_norm:
            return x, x_norm, x_v_norm_pre, x_v_norm_post, x_v
        return x, x_v
        


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout_rate: float,
        add_gated_x_attn: int, 
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True, 
                                       add_gated_x_attn=add_gated_x_attn,)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)      


    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, 
                xv: Optional[Tensor] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        
        x = x.to(xa.dtype)

        for layer, block in enumerate(self.blocks):
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache, xv=xv)
            
        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, dropout_rate: float, video: bool, 
                 video_model_path: str, av_hubert_path: str, prob_av: float, prob_a: float, av_hubert_encoder: bool,
                 av_fusion: str, add_adapter: bool, adapter_dim: int, add_gated_x_attn: int):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dropout_rate,
            video,
            video_model_path,
            av_hubert_path,
            prob_av,
            prob_a,
            av_hubert_encoder,
            av_fusion,
            add_adapter,
            adapter_dim,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dropout_rate,
            add_gated_x_attn,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        # all_heads = torch.zeros(
        #     self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        # )
        # all_heads[self.dims.n_text_layer // 2 :] = True
        # self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    # def set_alignment_heads(self, dump: bytes):
    #     array = np.frombuffer(
    #         gzip.decompress(base64.b85decode(dump)), dtype=bool
    #     ).copy()
    #     mask = torch.from_numpy(array).reshape(
    #         self.dims.n_text_layer, self.dims.n_text_head
    #     )
    #     self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
