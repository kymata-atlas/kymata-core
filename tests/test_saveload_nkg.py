from pathlib import Path
from tempfile import NamedTemporaryFile

from kymata.entities.expression import HexelExpressionSet, SensorExpressionSet
from kymata.io.nkg import save_expression_set, load_expression_set


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
            ed_saved_and_reloaded = load_expression_set(open_tf)

    if not already_existed:
        delete_dataset(sample_dataset)

    assert es_loaded_from_source == ed_saved_and_reloaded


def test_load_v0_1_nkg():
    es = load_expression_set(Path(Path(__file__).parent, "test-data", "version_0_1.nkg"))
    assert isinstance(es, HexelExpressionSet)
    assert len(es.functions) == 1
    assert es.functions == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.hexels) == 100
    assert es.left.shape == es.right.shape == (100, 10, 1)


def test_load_sensor_nkg():
    es = load_expression_set(Path(Path(__file__).parent, "test-data", "sensor.nkg"))
    assert isinstance(es, SensorExpressionSet)
    assert len(es.functions) == 1
    assert es.functions == ["test function"]
    assert len(es.latencies) == 10
    assert len(es.sensors) == 305
    assert es.scalp.shape == (305, 10, 1)