import re
from enum import StrEnum
from pathlib import Path

import yaml
from kymata.entities.rudimentary import Point2d


_layout_data_dir = Path(__file__).parent.parent / "data" / "sensor_locations"


class MEGLayout(StrEnum):
    """Represents supported MEG layouts."""
    Vectorview = "Vectorview"
    CTF_275    = "CTF275"
    KIT_AD     = "KIT-AD"


class EEGLayout(StrEnum):
    """Represents supported EEG layouts."""
    Easycap = "Easycap"


_lout_re = re.compile(
    r"^\d+\t"
    r"(?P<x>-?\d+\.\d+)\t"
    r"(?P<y>-?\d+\.\d+)\t"
    r"-?\d+\.\d+\t"
    r"-?\d+\.\d+\t"
    r"(?P<sensor>MEG \d+)$"
)


def _get_meg_sensor_xy_from_lout(filepath) -> dict[str, Point2d]:
    d = dict()
    with filepath.open("r") as layout_file:
        _ = layout_file.readline()  # First line is nothing
        for line in layout_file:
            if not line:
                continue  # Skip blank lines
            match = _lout_re.match(line)
            sensor = match.group("sensor")
            sensor = sensor.replace(" ", "")
            d[sensor] = Point2d(float(match.group("x")), float(match.group("y")))
    return d


def _get_meg_sensor_xy_vectorview() -> dict[str, Point2d]:
    return _get_meg_sensor_xy_from_lout(_layout_data_dir / "Vectorview-all.lout")


def _get_meg_sensor_xy_ctf275() -> dict[str, Point2d]:
    return _get_meg_sensor_xy_from_lout(_layout_data_dir / "CTF-275.lout")


def _get_meg_sensor_xy_kit_ad() -> dict[str, Point2d]:
    return _get_meg_sensor_xy_from_lout(_layout_data_dir / "KIT-AD.lout")


def get_meg_sensor_xy(layout: MEGLayout) -> dict[str, Point2d]:
    """
    Retrieve the 2D coordinates of MEG sensors of a given layout.

    This function reads the sensor locations from a predefined layout file and returns a dictionary mapping
    sensor names to their corresponding 2D coordinates.

    Returns:
    --------
    dict[str, Point2d]
        A dictionary where keys are sensor names (e.g., 'MEG1234') and values are Point2d objects representing
        the x and y coordinates of the sensors.
    """
    if layout == MEGLayout.Vectorview:
        return _get_meg_sensor_xy_vectorview()
    elif layout == MEGLayout.CTF_275:
        return _get_meg_sensor_xy_ctf275()
    elif layout == MEGLayout.KIT_AD:
        return _get_meg_sensor_xy_kit_ad()
    else:
        raise NotImplementedError()


def _get_eeg_sensor_xy_eeg1005() -> dict[str, Point2d]:
    with Path(_layout_data_dir, "EEG-layout-channel-mappings.yaml").open("r") as eeg_name_mapping_file:
        mapping = yaml.safe_load(eeg_name_mapping_file)
    mapping = {k.upper(): v.upper() for k, v in mapping.items()}
    d = dict()
    with Path(_layout_data_dir, "EEG1005.lay").open("r") as layout_file:
        for line in layout_file:
            parts = line.strip().split("\t")
            x = float(parts[1])
            y = float(parts[2])
            name = parts[-1].upper()
            d[name] = Point2d(x, y)
    our_sensor_d = {our_name: d[their_name] for our_name, their_name in mapping.items()}
    return our_sensor_d


def get_eeg_sensor_xy(layout: EEGLayout) -> dict[str, Point2d]:
    """
    Retrieve the 2D coordinates of EEG sensors of a given layout.

    This function reads the sensor locations and mappings from predefined layout and mapping files,
    then returns a dictionary mapping our sensor names to their corresponding 2D coordinates.

    Returns:
    --------
    dict[str, Point2d]
        A dictionary where keys are our sensor names and values are Point2d objects representing
        the x and y coordinates of the sensors.
    """
    if layout == EEGLayout.Easycap:
        return _get_eeg_sensor_xy_eeg1005()
    else:
        raise NotImplementedError()


def get_meg_sensors(layout: MEGLayout) -> set[str]:
    """Set of MEG sensor names."""
    return set(get_meg_sensor_xy(layout).keys())


def get_eeg_sensors(layout: EEGLayout) -> set[str]:
    """Set of EEG sensor names."""
    return set(get_eeg_sensor_xy(layout).keys())
