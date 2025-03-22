from matplotlib import pyplot as plt
from mne.io import Raw


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
