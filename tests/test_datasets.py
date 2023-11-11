from pathlib import Path

from kymata.datasets.sample import get_dataset_kymata_mirror_q3_2023, delete_dataset, get_dataset_gm_loudness


def test_download_and_delete_q3_2023_data_files():
    dataset = get_dataset_kymata_mirror_q3_2023(download=False)
    try:
        dataset.download()
        for filename in dataset.filenames:
            assert Path(dataset.path, filename).exists()
    finally:
        delete_dataset(dataset)
        for filename in dataset.filenames:
            assert not Path(dataset.path, filename).exists()


def test_download_and_delete_gm_loudness3_data_files():
    dataset = get_dataset_gm_loudness(download=False)
    try:
        dataset.download()
        for filename in dataset.filenames:
            assert Path(dataset.path, filename).exists()
    finally:
        delete_dataset(dataset)
        for filename in dataset.filenames:
            assert not Path(dataset.path, filename).exists()
