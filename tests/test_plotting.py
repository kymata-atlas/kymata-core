
from kymata.plot.plot import _get_best_ylim, _MAJOR_TICK_SIZE, _get_yticks


def test_best_best_ylim_returns_supplied_ylim():
    supplied_ylim = 1e-172
    data_y_min = 1e-250
    assert _get_best_ylim(ylim=supplied_ylim, data_y_min=data_y_min) == supplied_ylim


def test_best_data_ylim_is_multiple_of_major_tick_size():
    data_y_min = -51
    best_ylim = _get_best_ylim(ylim=None, data_y_min=data_y_min)
    assert -1 * best_ylim % _MAJOR_TICK_SIZE == 0


def test_small_data_gets_at_least_one_tick():
    data_y_min = 1
    ylim = _get_best_ylim(ylim=None, data_y_min=data_y_min)
    y_ticks = _get_yticks(ylim)
    assert len(y_ticks) >= 2
