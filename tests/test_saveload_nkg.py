"""
Tests the saving and loading of .nkg files.
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from kymata.entities.expression import HexelExpressionSet, SensorExpressionSet
from kymata.io.nkg2 import save_expression_set, load_expression_set, _load_data


def test_save_and_load_is_equal():
    from kymata.datasets.sample import TVLInsLoudnessOnlyDataset, delete_dataset

    sample_dataset = TVLInsLoudnessOnlyDataset(download=False)
    already_existed = sample_dataset.path.exists()
    sample_dataset.download()
    es_loaded_from_source = sample_dataset.to_expressionset()
    with NamedTemporaryFile(mode="wb", delete=False) as tf:
        save_expression_set(es_loaded_from_source, tf)
        tf.close()

        with open(tf.name, mode="rb") as open_tf:
            es_saved_and_reloaded = load_expression_set(open_tf)

    if not already_existed:
        delete_dataset(sample_dataset)

    assert es_loaded_from_source == es_saved_and_reloaded


def test_load_v0_1_nkg():
    from packaging import version

    v01_path = Path(Path(__file__).parent, "test-data", "version_0_1.nkg")
    v, _ = _load_data(v01_path)
    assert v == version.parse("0.1")
    es = load_expression_set(v01_path)
    assert isinstance(es, HexelExpressionSet)
    assert len(es.transforms) == 1
    assert es.transforms == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.hexels_left) == 100
    assert len(es.hexels_right) == 100
    assert es.left.shape == es.right.shape == (100, 10, 1)


def test_load_v0_2_hexel_nkg():
    from packaging import version

    v02_path = Path(Path(__file__).parent, "test-data", "version_0_2_hexel.nkg")
    v, _ = _load_data(v02_path)
    assert v == version.parse("0.2")
    es = load_expression_set(v02_path)
    assert isinstance(es, HexelExpressionSet)
    assert len(es.transforms) == 1
    assert es.transforms == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.hexels_left) == 100
    assert len(es.hexels_right) == 100
    assert es.left.shape == es.right.shape == (100, 10, 1)


def test_load_v0_2_sensor_nkg():
    es = load_expression_set(
        Path(Path(__file__).parent, "test-data", "version_0_2_sensor.nkg")
    )
    assert isinstance(es, SensorExpressionSet)
    assert len(es.transforms) == 1
    assert es.transforms == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.sensors) == 305
    assert es.scalp.shape == (305, 10, 1)


def test_load_v0_3_hexel_nkg():
    from packaging import version

    v03_path = Path(Path(__file__).parent, "test-data", "version_0_3_hexel.nkg")
    v, _ = _load_data(v03_path)
    assert v == version.parse("0.3")
    es = load_expression_set(v03_path)
    assert isinstance(es, HexelExpressionSet)
    assert len(es.transforms) == 1
    assert es.transforms == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.hexels_left) == 100
    assert len(es.hexels_right) == 100
    assert es.left.shape == es.right.shape == (100, 10, 1)


def test_load_v0_3_sensor_nkg():
    from packaging import version

    v03_path = Path(Path(__file__).parent, "test-data", "version_0_3_sensor.nkg")
    v, _ = _load_data(v03_path)
    assert v == version.parse("0.3")
    es = load_expression_set(v03_path)
    assert isinstance(es, SensorExpressionSet)
    assert len(es.transforms) == 1
    assert es.transforms == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.sensors) == 305
    assert es.scalp.shape == (305, 10, 1)


def test_load_v0_4_hexel_nkg():
    from packaging import version

    v04_path = Path(Path(__file__).parent, "test-data", "version_0_4_hexel.nkg")
    v, _ = _load_data(v04_path)
    assert v == version.parse("0.4")
    es = load_expression_set(v04_path)
    assert isinstance(es, HexelExpressionSet)
    assert len(es.transforms) == 1
    assert es.transforms == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.hexels_left) == 100
    assert len(es.hexels_right) == 100
    assert es.left.shape == es.right.shape == (100, 10, 1)


def test_load_v0_4_sensor_nkg():
    from packaging import version

    v04_path = Path(Path(__file__).parent, "test-data", "version_0_4_sensor.nkg")
    v, _ = _load_data(v04_path)
    assert v == version.parse("0.4")
    es = load_expression_set(v04_path)
    assert isinstance(es, SensorExpressionSet)
    assert len(es.transforms) == 1
    assert es.transforms == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.sensors) == 305
    assert es.scalp.shape == (305, 10, 1)


def test_load_v0_5_hexel_nkg():
    from packaging import version

    v05_path = Path(Path(__file__).parent, "test-data", "version_0_5_hexel.nkg")
    v, _ = _load_data(v05_path)
    assert v == version.parse("0.5")
    es = load_expression_set(v05_path)
    assert isinstance(es, HexelExpressionSet)
    assert len(es.transforms) == 1
    assert es.transforms == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.hexels_left) == 100
    assert len(es.hexels_right) == 100
    assert es.left.shape == es.right.shape == (100, 10, 1)


def test_load_v0_5_sensor_nkg():
    from packaging import version

    v05_path = Path(Path(__file__).parent, "test-data", "version_0_5_sensor.nkg")
    v, _ = _load_data(v05_path)
    assert v == version.parse("0.5")
    es = load_expression_set(v05_path)
    assert isinstance(es, SensorExpressionSet)
    assert len(es.transforms) == 1
    assert es.transforms == ["test transform"]
    assert len(es.latencies) == 200
    assert len(es.sensors) == 376
    assert es.scalp.shape == (376, 200, 1)


def test_load_multiple_files():
    v04_path = Path(Path(__file__).parent, "test-data", "version_0_4_sensor.nkg")
    v04_renamed_path = Path(
        Path(__file__).parent, "test-data", "version_0_4_sensor_renamed_function.nkg"
    )
    separate = load_expression_set(v04_path) + load_expression_set(v04_renamed_path)
    together = load_expression_set([v04_path, v04_renamed_path])
    assert len(separate.transforms) == 2
    assert separate == together


def test_multiple_files_empty_list():
    with pytest.raises(ValueError):
        load_expression_set([])
