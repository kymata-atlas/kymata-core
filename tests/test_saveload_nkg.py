from tempfile import NamedTemporaryFile

from kymata.entities.expression import ExpressionSet
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
