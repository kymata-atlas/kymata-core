from kymata.simulation.signal import white_noise


def test_white_noise_correct_length():
    assert len(white_noise(456, 10, 50)) == 456
