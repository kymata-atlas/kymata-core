import pytest
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


@pytest.fixture
def eeg_sensors():
    from kymata.plot.layouts import get_eeg_sensor_xy
    return list(get_eeg_sensor_xy().keys())


@pytest.fixture
def meg_sensors():
    from kymata.plot.layouts import get_meg_sensor_xy
    return list(get_meg_sensor_xy().keys())


@pytest.fixture
def all_sensors(eeg_sensors, meg_sensors):
    return eeg_sensors + meg_sensors


def test_eeg_correct_sensors(eeg_sensors):
    assert sorted(eeg_sensors) == sorted([
        "EEG001",
        "EEG002",
        "EEG003",
        "EEG004",
        "EEG005",
        "EEG006",
        "EEG007",
        "EEG008",
        "EEG009",
        "EEG000",
        "EEG011",
        "EEG012",
        "EEG013",
        "EEG014",
        "EEG015",
        "EEG016",
        "EEG017",
        "EEG019",
        "EEG020",
        "EEG021",
        "EEG022",
        "EEG023",
        "EEG024",
        "EEG025",
        "EEG026",
        "EEG027",
        "EEG030",
        "EEG031",
        "EEG032",
        "EEG033",
        "EEG034",
        "EEG035",
        "EEG036",
        "EEG037",
        "EEG038",
        "EEG040",
        "EEG041",
        "EEG042",
        "EEG043",
        "EEG044",
        "EEG045",
        "EEG046",
        "EEG047",
        "EEG048",
        "EEG049",
        "EEG050",
        "EEG052",
        "EEG053",
        "EEG054",
        "EEG055",
        "EEG056",
        "EEG057",
        "EEG058",
        "EEG059",
        "EEG060",
        "EEG062",
        "EEG063",
        "EEG064",
        "EEG018",
        "EEG028",
        "EEG029",
        "EEG039",
        "EEG051",
        "EEG061",
    ])
