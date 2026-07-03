import numpy as np

from kymata.preproc.sensor import _hilbert_envolope, _stretch_signal, fit_drift_delay

tau = 2 * np.pi


def test_hilbert_envelope_sine():
    fs = 1000
    t = np.arange(fs) / fs
    x = np.sin(tau * 20 * t)

    env = _hilbert_envolope(x)

    np.testing.assert_allclose(
        env.mean(),
        1.0,
        atol=1e-2,
    )


def test_hilbert_envelope_tracks_amplitude():
    fs = 1000
    t = np.arange(fs) / fs

    modulation = 1 + 0.5 * np.sin(2 * np.pi * 2 * t)
    carrier = np.sin(2 * np.pi * 100 * t)

    env = _hilbert_envolope(modulation * carrier)

    corr = np.corrcoef(env, modulation)[0, 1]

    assert corr > 0.99


def test_stretch_signal_identity():
    x = np.random.randn(1000)

    y = _stretch_signal(
        x,
        sample_rate=1000,
        stretch=1,
        delay=0,
    )

    np.testing.assert_allclose(x, y)


def test_stretch_signal_delay():
    x = np.zeros(100)
    x[20] = 1

    y = _stretch_signal(
        x,
        sample_rate=100,
        stretch=1,
        delay=0.1,
    )

    assert np.argmax(y) == 30
    
    
def test_fit_drift_delay_zero():
    fs = 1000

    x = np.random.randn(fs * 10)

    drift, delay = fit_drift_delay(
        x,
        x.copy(),
        fs,
        fs,
    )

    assert np.isclose(drift, 0)
    assert np.isclose(delay, 0)


def test_fit_drift_delay_known():
    fs = 1000

    stimulus = np.random.randn(fs * 30)

    true_drift = 2e-4
    true_delay = 7e-3

    misc = _stretch_signal(
        stimulus,
        sample_rate=fs,
        stretch=1 + true_drift,
        delay=true_delay,
    )

    drift, delay = fit_drift_delay(
        stimulus,
        misc,
        fs,
        fs,
        reference_drift=0,
        reference_delay=0,
        drift_range=0.001,
        delay_range=0.02,
        grid_n=81,
    )

    assert np.isclose(drift, true_drift)
    assert np.isclose(delay, true_delay)
