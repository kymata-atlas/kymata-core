import re
from pathlib import Path
from typing import NamedTuple

import yaml
from mne.io import Raw
from matplotlib import pyplot as plt


class Point2d(NamedTuple):
    x: float
    y: float


def get_meg_sensor_xy() -> dict[str, Point2d]:
    d = dict()
    layout_line_re = re.compile(
        r"^\d+\t"
        r"(?P<x>-?\d+\.\d+)\t"
        r"(?P<y>-?\d+\.\d+)\t"
        r"-?\d+\.\d+\t"
        r"-?\d+\.\d+\t"
        r"(?P<sensor>MEG \d+)$"
    )
    with Path(Path(__file__).parent.parent.parent, "kymata-toolbox-data", "sensor_locations", "Vectorview-all.lout").open("r") as layout_file:
        _ = layout_file.readline()  # First line is nothing
        for line in layout_file:
            if not line: 
                continue  # Skip blank lines
            match = layout_line_re.match(line)
            sensor = match.group("sensor")
            sensor = sensor.replace(" ", "")
            d[sensor] = Point2d(float(match.group("x")), float(match.group("y")))
    return d


def get_eeg_sensor_xy() -> dict[str, Point2d]:
    with Path(Path(__file__).parent.parent.parent, "kymata-toolbox-data", "sensor_locations",
              "EEG-layout-channel-mappings.yaml").open("r") as eeg_name_mapping_file:
        mapping = yaml.safe_load(eeg_name_mapping_file)
    mapping = {k.upper(): v.upper() for k, v in mapping.items()}
    d = dict()
    with Path(Path(__file__).parent.parent.parent, "kymata-toolbox-data", "sensor_locations",
              "EEG1005.lay").open("r") as layout_file:
        for line in layout_file:
            parts = line.strip().split("\t")
            x = float(parts[1])
            y = float(parts[2])
            name = parts[-1].upper()
            d[name] = Point2d(x, y)
    our_sensor_d = {
        our_name: d[their_name]
        for our_name, their_name in mapping.items()
    }
    return our_sensor_d


def plot_eeg_sensor_positions(raw_fif: Raw):
    """Plot Sensor positions"""
    fig = plt.figure()
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection='3d')
    raw_fif.plot_sensors(ch_type='eeg', axes=ax2d)
    raw_fif.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d')
    ax3d.view_init(azim=70, elev=15)
