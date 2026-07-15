from pathlib import Path

import numpy as np
from scipy.io import wavfile


def load_wav_as_floats(path: str | Path, mix_to_mono: bool = False) -> tuple[np.ndarray, int]:
    """
    Load a WAV file as float32.
    
    If mix_to_mono is True, all channels will be averaged to get the mono channel

    Returns:
        (audio, sample_rate):
            audio: numpy array with dtype float32 and shape:
              - (samples,) for mono
              - (samples, channels) for stereo and other multi-channel
            sample_rate: float (sample rate in Hz)
    """
    sample_rate, audio = wavfile.read(path)

    if np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32, copy=False)
    elif np.issubdtype(audio.dtype, np.signedinteger):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif np.issubdtype(audio.dtype, np.unsignedinteger):
        info = np.iinfo(audio.dtype)
        audio = (audio.astype(np.float32) - (info.max + 1) / 2) / (
            (info.max + 1) / 2
        )
    else:
        raise TypeError(f"Unsupported WAV dtype: {audio.dtype}")
    
    if mix_to_mono and audio.ndim > 1:
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        else:
            raise ValueError(f"Too many dimensions in audio array (expected 2, got {audio.ndim})")

    return audio, sample_rate
