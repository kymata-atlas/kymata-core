from pathlib import Path
from typing import Optional

from mne.datasets import fetch_fsaverage

from kymata.datasets.sample import SampleDataset
from kymata.io.file import PathType


class FSAverageDataset(SampleDataset):
    def __init__(
        self,
        data_root: Optional[PathType] = None,
        download: bool = True,
    ):
        from mne.datasets._fsaverage.base import FSAVERAGE_MANIFEST_PATH

        # Filenames are handled by mne:
        filenames = []
        with Path(FSAVERAGE_MANIFEST_PATH, "root.txt").open("r") as fsaverage_manifest:
            for line in fsaverage_manifest:
                filenames.append(line.strip())
        with Path(FSAVERAGE_MANIFEST_PATH, "bem.txt").open("r") as bem_manifest:
            for line in bem_manifest:
                filenames.append("fsaverage/" + line.strip())

        super().__init__(
            name="fsaverage",
            data_root=data_root,
            remote_root=None,  # Remote paths are handled by mne
            download=download,
            filenames=filenames,
        )

    def download(self):
        print(f"Downloading dataset: {self.name}")
        self.path.mkdir(exist_ok=True)
        _dir = fetch_fsaverage(subjects_dir=str(self.path), verbose=__debug__)
