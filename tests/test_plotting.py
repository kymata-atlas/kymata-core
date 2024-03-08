from pathlib import Path

from numpy import array, array_equal

from kymata.datasets.sample import delete_dataset
from kymata.plot.plot import _get_best_ylim, _MAJOR_TICK_SIZE, _get_yticks, _get_xticks, expression_plot



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


def test_get_x_ticks_standard():
    x_ticks = _get_xticks((-200, 800))
    assert array_equal(x_ticks, array([-200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800]))


def test_get_x_ticks_smaller():
    x_ticks = _get_xticks((-100, 700))
    assert array_equal(x_ticks, array([-100, 0, 100, 200, 300, 400, 500, 600, 700]))


def test_get_x_ticks_non_multiples():
    x_ticks = _get_xticks((-150, 750))
    assert array_equal(x_ticks, array([-100, 0, 100, 200, 300, 400, 500, 600, 700]))


def test_expression_plot_no_error():
    from kymata.datasets.sample import TVLInsLoudnessOnlyDataset
    dataset = TVLInsLoudnessOnlyDataset(download=False)
    try:
        dataset.download()

        expression_plot(dataset.to_expressionset())

    finally:
        delete_dataset(dataset)
