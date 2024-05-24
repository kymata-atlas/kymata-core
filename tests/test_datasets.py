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
    from kymata.datasets.sample import TVLInsLoudnessOnlyDataset
    dataset = TVLInsLoudnessOnlyDataset(download=False)
    try:
        dataset.download()
        for filename in dataset.filenames:
            assert Path(dataset.path, filename).exists()
    finally:
        delete_dataset(dataset)
        for filename in dataset.filenames:
            assert not Path(dataset.path, filename).exists()


def test_download_and_delete_fsaverage_surfaces():
    from kymata.datasets.fsaverage import FSAverageDataset

    fsaverage_dataset = FSAverageDataset()
    try:
        fsaverage_dataset.download()
        for filename in fsaverage_dataset.filenames:
            assert Path(fsaverage_dataset.path, filename).exists()
    finally:
        delete_dataset(fsaverage_dataset)
        for filename in fsaverage_dataset.filenames:
            assert not Path(fsaverage_dataset.path, filename).exists()
