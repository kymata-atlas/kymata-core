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
    """
    Retrieve the 2D coordinates of MEG sensors.

    This function reads the sensor locations from a predefined layout file and returns a dictionary mapping
    sensor names to their corresponding 2D coordinates.

    Returns:
    --------
    dict[str, Point2d]
        A dictionary where keys are sensor names (e.g., 'MEG1234') and values are Point2d objects representing
        the x and y coordinates of the sensors.

    Notes:
    ------
    The function expects the layout file to be located at 'kymata-core-data/sensor_locations/Vectorview-all.lout'.
    """

    d = dict()
    layout_line_re = re.compile(
        r"^\d+\t"
        r"(?P<x>-?\d+\.\d+)\t"
        r"(?P<y>-?\d+\.\d+)\t"
        r"-?\d+\.\d+\t"
        r"-?\d+\.\d+\t"
        r"(?P<sensor>MEG \d+)$"
    )
    with Path(
        Path(__file__).parent.parent.parent,
        "kymata-core-data",
        "sensor_locations",
        "Vectorview-all.lout",
    ).open("r") as layout_file:
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
    """
    Retrieve the 2D coordinates of EEG sensors.

    This function reads the sensor locations and mappings from predefined layout and mapping files,
    then returns a dictionary mapping our sensor names to their corresponding 2D coordinates.

    Returns:
    --------
    dict[str, Point2d]
        A dictionary where keys are our sensor names and values are Point2d objects representing
        the x and y coordinates of the sensors.

    Notes:
    ------
    The function expects the layout file to be located at 'kymata-core-data/sensor_locations/EEG1005.lay'
    and the mapping file to be located at 'kymata-core-data/sensor_locations/EEG-layout-channel-mappings.yaml'.
    """
    with Path(
        Path(__file__).parent.parent.parent,
        "kymata-core-data",
        "sensor_locations",
        "EEG-layout-channel-mappings.yaml",
    ).open("r") as eeg_name_mapping_file:
        mapping = yaml.safe_load(eeg_name_mapping_file)
    mapping = {k.upper(): v.upper() for k, v in mapping.items()}
    d = dict()
    with Path(
        Path(__file__).parent.parent.parent,
        "kymata-core-data",
        "sensor_locations",
        "EEG1005.lay",
    ).open("r") as layout_file:
        for line in layout_file:
            parts = line.strip().split("\t")
            x = float(parts[1])
            y = float(parts[2])
            name = parts[-1].upper()
            d[name] = Point2d(x, y)
    our_sensor_d = {our_name: d[their_name] for our_name, their_name in mapping.items()}
    return our_sensor_d


def get_meg_sensors() -> set[str]:
    """Set of MEG sensor names."""
    return set(get_meg_sensor_xy().keys())


def get_eeg_sensors() -> set[str]:
    """Set of EEG sensor names."""
    return set(get_eeg_sensor_xy().keys())


def plot_eeg_sensor_positions(raw_fif: Raw):
    """
    Plot the positions of EEG sensors in 2D and 3D views.

    This function generates a figure with two subplots: a 2D view and a 3D view of the EEG sensor positions.

    Parameters:
    -----------
    raw_fif : Raw
        The raw FIF file containing EEG data and sensor locations.

    Notes:
    ------
    The 3D plot is initialized with an azimuth angle of 70 degrees and an elevation angle of 15 degrees for better visualization.
    """

    fig = plt.figure()
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection="3d")
    raw_fif.plot_sensors(ch_type="eeg", axes=ax2d)
    raw_fif.plot_sensors(ch_type="eeg", axes=ax3d, kind="3d")
    ax3d.view_init(azim=70, elev=15)
