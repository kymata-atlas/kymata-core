# Tests written with some genai help

import numpy as np
from scipy.io import wavfile

from kymata.io.audio import load_wav_as_floats


def test_load_wav_as_floats_mono(tmp_path):
    sr = 16000
    pcm = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)

    path = tmp_path / "mono.wav"
    wavfile.write(path, sr, pcm)

    expected = pcm.astype(np.float32) / np.iinfo(np.int16).max

    loaded, loaded_sr = load_wav_as_floats(path)

    assert loaded.dtype == np.float32
    assert loaded_sr == sr
    np.testing.assert_allclose(loaded, expected, atol=1e-7)


def test_load_wav_as_floats_stereo(tmp_path):
    sr = 44100
    pcm = np.array(
        [[1, 2], [3, 4], [5, 6]],
        dtype=np.int16,
    )

    path = tmp_path / "stereo.wav"
    wavfile.write(path, sr, pcm)

    expected = pcm.astype(np.float32) / np.iinfo(np.int16).max

    loaded, loaded_sr = load_wav_as_floats(path)

    assert loaded.dtype == np.float32
    assert loaded_sr == sr
    np.testing.assert_allclose(loaded, expected, atol=1e-7)


def test_load_wav_as_floats_float32(tmp_path):
    sr = 22050
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)

    path = tmp_path / "float.wav"
    wavfile.write(path, sr, audio)

    loaded, loaded_sr = load_wav_as_floats(path)

    assert loaded.dtype == np.float32
    assert loaded_sr == sr
    np.testing.assert_allclose(loaded, audio, atol=1e-7)


def test_load_wav_as_floats_stereo_preserves_channels(tmp_path):
    sr = 44100
    pcm = np.array(
        [[1000, -1000],
         [2000, -2000],
         [3000, -3000]],
        dtype=np.int16,
    )

    path = tmp_path / "stereo.wav"
    wavfile.write(path, sr, pcm)

    loaded, loaded_sr = load_wav_as_floats(path)

    expected = pcm.astype(np.float32) / np.iinfo(np.int16).max

    assert loaded_sr == sr
    assert loaded.dtype == np.float32
    assert loaded.shape == (3, 2)
    np.testing.assert_allclose(loaded, expected)


def test_load_wav_as_floats_mix_to_mono(tmp_path):
    sr = 44100
    pcm = np.array(
        [[1000, -1000],
         [2000, 2000],
         [-3000, 1000]],
        dtype=np.int16,
    )

    path = tmp_path / "stereo.wav"
    wavfile.write(path, sr, pcm)

    loaded, loaded_sr = load_wav_as_floats(path, mix_to_mono=True)

    expected = (
        pcm.astype(np.float32) / np.iinfo(np.int16).max
    ).mean(axis=1)

    assert loaded_sr == sr
    assert loaded.dtype == np.float32
    assert loaded.shape == (3,)
    np.testing.assert_allclose(loaded, expected)


def test_load_wav_as_floats_mix_to_mono_is_noop_for_mono(tmp_path):
    sr = 16000
    pcm = np.array([1000, -1000, 2000], dtype=np.int16)

    path = tmp_path / "mono.wav"
    wavfile.write(path, sr, pcm)

    loaded, _ = load_wav_as_floats(path, mix_to_mono=True)

    expected = pcm.astype(np.float32) / np.iinfo(np.int16).max

    assert loaded.shape == (3,)
    np.testing.assert_allclose(loaded, expected)
