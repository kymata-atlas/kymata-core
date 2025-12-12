from pathlib import Path

import pytest
from numpy import array, array_equal

from kymata.datasets.sample import delete_dataset
from kymata.io.layouts import SensorLayout, MEGLayout, EEGLayout
from kymata.io.nkg import load_expression_set
from kymata.plot.color import gradient_color_dict
from kymata.plot.expression import (
    _get_best_ylim,
    _MAJOR_TICK_SIZE,
    _get_yticks,
    _get_xticks,
    expression_plot,
)


test_layout = SensorLayout(meg=MEGLayout.Vectorview, eeg=EEGLayout.Easycap)


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
    assert array_equal(
        x_ticks, array([-200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800])
    )


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
    from kymata.io.layouts import get_eeg_sensor_xy

    return list(get_eeg_sensor_xy(test_layout.eeg).keys())


@pytest.fixture
def meg_sensors():
    from kymata.io.layouts import get_meg_sensor_xy

    return list(get_meg_sensor_xy(test_layout.meg).keys())


def test_eeg_correct_sensors(eeg_sensors):
    assert sorted(eeg_sensors) == sorted(
        [
            "EEG001",
            "EEG002",
            "EEG003",
            "EEG004",
            "EEG005",
            "EEG006",
            "EEG007",
            "EEG008",
            "EEG009",
            "EEG010",
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
        ]
    )


def test_meg_correct_sensors(meg_sensors):
    assert sorted(meg_sensors) == sorted(
        [
            "MEG0113",
            "MEG0112",
            "MEG0111",
            "MEG0122",
            "MEG0123",
            "MEG0121",
            "MEG0132",
            "MEG0133",
            "MEG0131",
            "MEG0143",
            "MEG0142",
            "MEG0141",
            "MEG0213",
            "MEG0212",
            "MEG0211",
            "MEG0222",
            "MEG0223",
            "MEG0221",
            "MEG0232",
            "MEG0233",
            "MEG0231",
            "MEG0243",
            "MEG0242",
            "MEG0241",
            "MEG0313",
            "MEG0312",
            "MEG0311",
            "MEG0322",
            "MEG0323",
            "MEG0321",
            "MEG0333",
            "MEG0332",
            "MEG0331",
            "MEG0343",
            "MEG0342",
            "MEG0341",
            "MEG0413",
            "MEG0412",
            "MEG0411",
            "MEG0422",
            "MEG0423",
            "MEG0421",
            "MEG0432",
            "MEG0433",
            "MEG0431",
            "MEG0443",
            "MEG0442",
            "MEG0441",
            "MEG0513",
            "MEG0512",
            "MEG0511",
            "MEG0523",
            "MEG0522",
            "MEG0521",
            "MEG0532",
            "MEG0533",
            "MEG0531",
            "MEG0542",
            "MEG0543",
            "MEG0541",
            "MEG0613",
            "MEG0612",
            "MEG0611",
            "MEG0622",
            "MEG0623",
            "MEG0621",
            "MEG0633",
            "MEG0632",
            "MEG0631",
            "MEG0642",
            "MEG0643",
            "MEG0641",
            "MEG0713",
            "MEG0712",
            "MEG0711",
            "MEG0723",
            "MEG0722",
            "MEG0721",
            "MEG0733",
            "MEG0732",
            "MEG0731",
            "MEG0743",
            "MEG0742",
            "MEG0741",
            "MEG0813",
            "MEG0812",
            "MEG0811",
            "MEG0822",
            "MEG0823",
            "MEG0821",
            "MEG0913",
            "MEG0912",
            "MEG0911",
            "MEG0923",
            "MEG0922",
            "MEG0921",
            "MEG0932",
            "MEG0933",
            "MEG0931",
            "MEG0942",
            "MEG0943",
            "MEG0941",
            "MEG1013",
            "MEG1012",
            "MEG1011",
            "MEG1023",
            "MEG1022",
            "MEG1021",
            "MEG1032",
            "MEG1033",
            "MEG1031",
            "MEG1043",
            "MEG1042",
            "MEG1041",
            "MEG1112",
            "MEG1113",
            "MEG1111",
            "MEG1123",
            "MEG1122",
            "MEG1121",
            "MEG1133",
            "MEG1132",
            "MEG1131",
            "MEG1142",
            "MEG1143",
            "MEG1141",
            "MEG1213",
            "MEG1212",
            "MEG1211",
            "MEG1223",
            "MEG1222",
            "MEG1221",
            "MEG1232",
            "MEG1233",
            "MEG1231",
            "MEG1243",
            "MEG1242",
            "MEG1241",
            "MEG1312",
            "MEG1313",
            "MEG1311",
            "MEG1323",
            "MEG1322",
            "MEG1321",
            "MEG1333",
            "MEG1332",
            "MEG1331",
            "MEG1342",
            "MEG1343",
            "MEG1341",
            "MEG1412",
            "MEG1413",
            "MEG1411",
            "MEG1423",
            "MEG1422",
            "MEG1421",
            "MEG1433",
            "MEG1432",
            "MEG1431",
            "MEG1442",
            "MEG1443",
            "MEG1441",
            "MEG1512",
            "MEG1513",
            "MEG1511",
            "MEG1522",
            "MEG1523",
            "MEG1521",
            "MEG1533",
            "MEG1532",
            "MEG1531",
            "MEG1543",
            "MEG1542",
            "MEG1541",
            "MEG1613",
            "MEG1612",
            "MEG1611",
            "MEG1622",
            "MEG1623",
            "MEG1621",
            "MEG1632",
            "MEG1633",
            "MEG1631",
            "MEG1643",
            "MEG1642",
            "MEG1641",
            "MEG1713",
            "MEG1712",
            "MEG1711",
            "MEG1722",
            "MEG1723",
            "MEG1721",
            "MEG1732",
            "MEG1733",
            "MEG1731",
            "MEG1743",
            "MEG1742",
            "MEG1741",
            "MEG1813",
            "MEG1812",
            "MEG1811",
            "MEG1822",
            "MEG1823",
            "MEG1821",
            "MEG1832",
            "MEG1833",
            "MEG1831",
            "MEG1843",
            "MEG1842",
            "MEG1841",
            "MEG1912",
            "MEG1913",
            "MEG1911",
            "MEG1923",
            "MEG1922",
            "MEG1921",
            "MEG1932",
            "MEG1933",
            "MEG1931",
            "MEG1943",
            "MEG1942",
            "MEG1941",
            "MEG2013",
            "MEG2012",
            "MEG2011",
            "MEG2023",
            "MEG2022",
            "MEG2021",
            "MEG2032",
            "MEG2033",
            "MEG2031",
            "MEG2042",
            "MEG2043",
            "MEG2041",
            "MEG2113",
            "MEG2112",
            "MEG2111",
            "MEG2122",
            "MEG2123",
            "MEG2121",
            "MEG2133",
            "MEG2132",
            "MEG2131",
            "MEG2143",
            "MEG2142",
            "MEG2141",
            "MEG2212",
            "MEG2213",
            "MEG2211",
            "MEG2223",
            "MEG2222",
            "MEG2221",
            "MEG2233",
            "MEG2232",
            "MEG2231",
            "MEG2242",
            "MEG2243",
            "MEG2241",
            "MEG2312",
            "MEG2313",
            "MEG2311",
            "MEG2323",
            "MEG2322",
            "MEG2321",
            "MEG2332",
            "MEG2333",
            "MEG2331",
            "MEG2343",
            "MEG2342",
            "MEG2341",
            "MEG2412",
            "MEG2413",
            "MEG2411",
            "MEG2423",
            "MEG2422",
            "MEG2421",
            "MEG2433",
            "MEG2432",
            "MEG2431",
            "MEG2442",
            "MEG2443",
            "MEG2441",
            "MEG2512",
            "MEG2513",
            "MEG2511",
            "MEG2522",
            "MEG2523",
            "MEG2521",
            "MEG2533",
            "MEG2532",
            "MEG2531",
            "MEG2543",
            "MEG2542",
            "MEG2541",
            "MEG2612",
            "MEG2613",
            "MEG2611",
            "MEG2623",
            "MEG2622",
            "MEG2621",
            "MEG2633",
            "MEG2632",
            "MEG2631",
            "MEG2642",
            "MEG2643",
            "MEG2641",
        ]
    )


def test_eeg_left_right_medial_count(eeg_sensors):
    from kymata.plot.sensor import get_sensor_left_right_assignment
    left_chans, right_chans = get_sensor_left_right_assignment(test_layout)
    top_chans = left_chans
    bottom_chans = right_chans
    both_chans = top_chans & bottom_chans
    top_chans -= both_chans
    bottom_chans -= both_chans

    top_eeg = [e for e in eeg_sensors if e in top_chans]
    bottom_eeg = [e for e in eeg_sensors if e in bottom_chans]
    both_eeg = [e for e in eeg_sensors if e in both_chans]

    assert len(both_eeg) == 10
    assert len(top_eeg) == 27
    assert len(bottom_eeg) == 27


def test_expression_set_plot_with_explicit_colour_for_hidden_transform():
    es = load_expression_set(
        Path(Path(__file__).parent, "test-data", "many_functions.nkg")
    )
    expression_plot(
        es,
        color=gradient_color_dict(
            [
                "d_IL1",
                "d_IL2",
                "d_IL3",
                "d_IL4",
                "d_IL5",
                "d_IL6",
                "d_IL7",
                "d_IL8",
                "d_IL9",
            ],
            start_color="orange",
            stop_color="yellow",
        ),
        show_only=["d_IL1"],
        paired_axes=False,
    )
