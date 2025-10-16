from pathlib import Path
import pytest
import os

from kymata.datasets.sample import delete_dataset

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Test only when run locally - otherwise github will download 0.5GB each time we push",
)
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


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Test only when run locally - otherwise github will download files each time we push",
)
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


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Test only when run locally - otherwise github will download files each time we push",
)
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

def test_fsaverage_dataset_init_no_download():
    """
    Test the initialization of FSAverageDataset when download is False.
    This should correctly populate filenames without attempting a download.
    """
    from kymata.datasets.fsaverage import FSAverageDataset
    from mne.datasets._fsaverage.base import FSAVERAGE_MANIFEST_PATH

    # We need to mock fetch_fsaverage to prevent actual download attempts
    # and also ensure the manifest files exist for filename population.
    # For this test, we will create dummy manifest files.
    temp_manifest_path = Path("temp_mne_manifest")
    temp_manifest_path.mkdir(exist_ok=True)

    original_manifest_path = FSAVERAGE_MANIFEST_PATH
    try:
        # Create dummy root.txt
        with open(temp_manifest_path / "root.txt", "w") as f:
            f.write("surf/rh.white\n")
            f.write("surf/lh.white\n")

        # Create dummy bem.txt
        with open(temp_manifest_path / "bem.txt", "w") as f:
            f.write("fsaverage-src.fif\n")

        # Temporarily patch FSAVERAGE_MANIFEST_PATH
        import mne.datasets._fsaverage.base
        mne.datasets._fsaverage.base.FSAVERAGE_MANIFEST_PATH = str(temp_manifest_path)

        # Initialize the dataset with download=False
        dataset = FSAverageDataset(download=False)

        # Assert that filenames are populated correctly
        expected_filenames = [
            "surf/rh.white",
            "surf/lh.white",
            "fsaverage/fsaverage-src.fif",
        ]
        assert sorted(dataset.filenames) == sorted(expected_filenames)

        # Assert that download method was not called (implicitly by checking no files are downloaded)
        # This is harder to test directly without mocking, but the purpose is to check init.
        # The lack of a download attempt means the `download` method is not called during init.

    finally:
        # Clean up dummy manifest files and directory
        if (temp_manifest_path / "root.txt").exists():
            (temp_manifest_path / "root.txt").unlink()
        if (temp_manifest_path / "bem.txt").exists():
            (temp_manifest_path / "bem.txt").unlink()
        if temp_manifest_path.exists():
            temp_manifest_path.rmdir()
        # Restore original FSAVERAGE_MANIFEST_PATH
        import mne.datasets._fsaverage.base
        mne.datasets._fsaverage.base.FSAVERAGE_MANIFEST_PATH = original_manifest_path
