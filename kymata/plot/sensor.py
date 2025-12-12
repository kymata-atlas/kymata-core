from matplotlib import pyplot as plt
from mne.io import Raw

from kymata.entities.datatypes import Channel
from kymata.entities.expression import ExpressionSet, ExpressionPoint, SensorExpressionSet, HexelExpressionSet
from kymata.io.layouts import SensorLayout, get_meg_sensor_xy, get_eeg_sensor_xy, get_meg_sensors, get_eeg_sensors


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


def get_sensor_left_right_assignment(layout: SensorLayout) -> tuple[set, set]:
    """
    Gets the left/right assignment of sensors to axes from a layout
    """
    left_sensors, right_sensors = [], []
    if layout.meg is not None:
        left_sensors.extend([sensor for sensor, (x, y) in get_meg_sensor_xy(layout.meg).items() if x <= 0.5])
        right_sensors.extend([sensor for sensor, (x, y) in get_meg_sensor_xy(layout.meg).items() if x >= 0.5])
    if layout.eeg is not None:
        left_sensors.extend([sensor for sensor, (x, y) in get_eeg_sensor_xy(layout.eeg).items() if x <= 0])
        right_sensors.extend([sensor for sensor, (x, y) in get_eeg_sensor_xy(layout.eeg).items() if x >= 0])
    return set(left_sensors), set(right_sensors)


def restrict_sensors_by_type(
    expression_set: ExpressionSet,
    best_transforms: tuple[list[ExpressionPoint], ...],
    show_only_sensors: str | None,
) -> set[Channel]:
    """
    Restrict to specific sensor type if requested.
    Does nothing to HexelExpressionSets.

    Args:
        expression_set:
        best_transforms:
        show_only_sensors:

    Returns:

    """
    if show_only_sensors is not None:
        if isinstance(expression_set, SensorExpressionSet):
            if show_only_sensors == "meg":
                chosen_channels = get_meg_sensors()
            elif show_only_sensors == "eeg":
                chosen_channels = get_eeg_sensors()
            else:
                raise NotImplementedError()
        else:
            raise ValueError("`show_only_sensors` only valid with sensor data.")
    else:
        if isinstance(expression_set, SensorExpressionSet):
            # All sensors
            chosen_channels = {
                ep.channel
                for best_transforms_each_ax in best_transforms
                for ep in best_transforms_each_ax
            }
        elif isinstance(expression_set, HexelExpressionSet):
            # All hexels
            chosen_channels = {
                ep.channel
                for best_transforms_each_ax in best_transforms
                for ep in best_transforms_each_ax
            }
        else:
            raise NotImplementedError()
    return chosen_channels
