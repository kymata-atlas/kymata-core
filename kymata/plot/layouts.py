import re
from pathlib import Path
from typing import NamedTuple


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
    with Path(Path(__file__).parent.parent, "config", "Vectorview-all.lout").open("r") as layout_file:
        _ = layout_file.readline()  # First line is nothing
        for line in layout_file:
            if not line: continue  # Skip blank lines
            match = layout_line_re.match(line)
            sensor = match.group("sensor")
            sensor = sensor.replace(" ", "")
            d[sensor] = Point2d(float(match.group("x")), float(match.group("y")))
    return d


def eeg_sensors() -> list[str]:
    return [f"EEG{i:03}" for i in range(1, 65)]
