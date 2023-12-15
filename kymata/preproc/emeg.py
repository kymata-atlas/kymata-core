from pathlib import Path

from mne.io import Raw
from mne.preprocessing import find_bad_channels_maxwell
from pandas import DataFrame, Index
from numpy import nanmin


def apply_automatic_bad_channel_detection(raw_fif_data: Raw, machine_used: str, plot: bool = True):
    """Apply Automatic Bad Channel Detection"""
    raw_check = raw_fif_data.copy()

    fine_cal_file = Path(Path(__file__).parent.parent,
                         'config', 'cbu_specific_files',
                         'sss_cal' + machine_used + '.dat')
    crosstalk_file = Path(Path(__file__).parent.parent,
                          'config', 'cbu_specific_files',
                          'ct_sparse' + machine_used + '.fif')

    auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
        raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
        return_scores=True, verbose=True)
    print(auto_noisy_chs)
    print(auto_flat_chs)

    bads = raw_fif_data.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw_fif_data.info['bads'] = bads

    if plot:
        _plot_bad_chans(auto_scores)

    return raw_fif_data


def _plot_bad_chans(auto_scores):
    import seaborn as sns
    from matplotlib import pyplot as plt

    # Only select the data for gradiometer channels.
    ch_type = 'grad'
    ch_subset = auto_scores['ch_types'] == ch_type
    ch_names = auto_scores['ch_names'][ch_subset]
    scores = auto_scores['scores_noisy'][ch_subset]
    limits = auto_scores['limits_noisy'][ch_subset]
    bins = auto_scores['bins']  # The windows that were evaluated.

    # We will label each segment by its start and stop time, with up to 3
    # digits before and 3 digits after the decimal place (1 ms precision).
    bin_labels = [f'{start:3.3f} – {stop:3.3f}'
                  for start, stop in bins]

    # We store the data in a Pandas DataFrame. The seaborn heatmap function
    # we will call below will then be able to automatically assign the correct
    # labels to all axes.
    data_to_plot = DataFrame(data=scores,
                             columns=Index(bin_labels, name='Time (s)'),
                             index=Index(ch_names, name='Channel'))

    # First, plot the "raw" scores.
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Automated noisy channel detection: {ch_type}',
                 fontsize=16, fontweight='bold')
    sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
                ax=ax[0])
    [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
     for x in range(1, len(bins))]
    ax[0].set_title('All Scores', fontweight='bold')

    # Now, adjust the color range to highlight segments that exceeded the limit.
    sns.heatmap(data=data_to_plot,
                vmin=nanmin(limits),  # bads in input data have NaN limits
                cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
    [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
     for x in range(1, len(bins))]
    ax[1].set_title('Scores > Limit', fontweight='bold')

    # The figure title should not overlap with the subplots.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Replace the word “noisy” with “flat”, and replace
    # vmin=nanmin(limits) with vmax=nanmax(limits) to print flat channels
