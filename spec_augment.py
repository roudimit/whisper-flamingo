"""Spec Augment module for preprocessing i.e., data augmentation"""
# Modified from ESPnet, takes into account max audio time
import random
import numpy

def freq_mask(x, audio_frames, F=30, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray x: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace and x.flags.writeable:
        cloned = x
    else:
        cloned = x.copy()

    num_mel_channels = cloned.shape[1]
    len_spectro = audio_frames
    fs = numpy.random.randint(0, F, size=(n_mask, 2))

    for f, mask_end in fs:
        f_zero = random.randrange(0, num_mel_channels - f)
        mask_end += f_zero

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            continue

        if replace_with_zero:
            cloned[:audio_frames, f_zero:mask_end] = 0
        else:
            cloned[:audio_frames, f_zero:mask_end] = cloned.mean()
    return cloned

def time_mask(spec, audio_frames, T=40, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace and spec.flags.writeable:
        cloned = spec
    else:
        cloned = spec.copy()
    len_spectro = audio_frames
    ts = numpy.random.randint(0, T, size=(n_mask, 2))
    for t, mask_end in ts:
        # avoid randint range error
        if len_spectro - t <= 0:
            continue
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            continue

        mask_end += t_zero
        if replace_with_zero:
            cloned[t_zero:mask_end] = 0
        else:
            cloned[t_zero:mask_end] = cloned.mean()
    return cloned

def spec_augment(
    x,
    audio_frames,
    resize_mode="PIL",
    max_time_warp=80,
    max_freq_width=27,
    n_freq_mask=2,
    max_time_width=100,
    n_time_mask=2,
    inplace=True,
    replace_with_zero=True,
):
    """spec agument

    apply random time warping and time/freq masking
    default setting is based on LD (Librispeech double) in Table 2
        https://arxiv.org/pdf/1904.08779.pdf

    :param numpy.ndarray x: (time, freq)
    :param str resize_mode: "PIL" (fast, nondifferentiable) or "sparse_image_warp"
        (slow, differentiable)
    :param int max_time_warp: maximum frames to warp the center frame in spectrogram (W)
    :param int freq_mask_width: maximum width of the random freq mask (F)
    :param int n_freq_mask: the number of the random freq mask (m_F)
    :param int time_mask_width: maximum width of the random time mask (T)
    :param int n_time_mask: the number of the random time mask (m_T)
    :param bool inplace: overwrite intermediate array
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    assert isinstance(x, numpy.ndarray)
    assert x.ndim == 2
    # Note: getting an error with time warp, so removed it
    x = freq_mask(
        x,
        audio_frames,
        max_freq_width,
        n_freq_mask,
        inplace=inplace,
        replace_with_zero=replace_with_zero,
    )
    x = time_mask(
        x,
        audio_frames,
        max_time_width,
        n_time_mask,
        inplace=inplace,
        replace_with_zero=replace_with_zero,
    )
    return x
