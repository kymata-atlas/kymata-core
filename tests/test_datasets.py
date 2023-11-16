from pathlib import Path

from kymata.datasets.sample import delete_dataset


def test_download_and_delete_q3_2023_data_files():
    from kymata.datasets.sample import KymataMirror2023Q3Dataset
    dataset = KymataMirror2023Q3Dataset(download=False)
    try:
        dataset.download()
        for filename in dataset.filenames:
            assert Path(dataset.path, filename).exists()
    finally:
        delete_dataset(dataset)
        for filename in dataset.filenames:
            assert not Path(dataset.path, filename).exists()


def test_download_and_delete_gm_loudness3_data_files():
    from kymata.datasets.sample import GMLoudnessDataset
    dataset = GMLoudnessDataset(download=False)
    try:
        dataset.download()
        for filename in dataset.filenames:
            assert Path(dataset.path, filename).exists()
    finally:
        delete_dataset(dataset)
        for filename in dataset.filenames:
            assert not Path(dataset.path, filename).exists()
