"""
====================================================
Read a forward operator and display sensitivity maps
====================================================

Forward solutions can be read using read_forward_solution in Python.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os
import numpy as np
import sys
sys.path.append('/imaging/local/software/mne_python/v0.9full')
sys.path.append('/imaging/local/software/python_packages/pysurfer/v0.4')


import mne
import matplotlib.pyplot as plt

print(__doc__)

fname = '/imaging/at03/NKG_Code_output/Version5/DATASET_3-02_tactile_toes/3-sensor-data/forward-models/meg15_0537_1_ico-5-3L-fwd.fif'
subjects_dir = '/imaging/at03/NKG_Data_Sets/DATASET_1-01_visual-only/mne_subjects_dir'

fwd = mne.read_forward_solution(fname, surf_ori=True)
leadfield = fwd['sol']['data']

print("Leadfield size : %d x %d" % leadfield.shape)

###############################################################################
# Show gain matrix a.k.a. leadfield matrix with sensitivity map

picks_meg = mne.pick_types(fwd['info'], meg=True, eeg=False)
picks_eeg = mne.pick_types(fwd['info'], meg=False, eeg=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Lead field matrix (500 dipoles only)', fontsize=14)

for ax, picks, ch_type in zip(axes, [picks_meg, picks_eeg], ['meg', 'eeg']):
    im = ax.imshow(leadfield[picks, :500], origin='lower', aspect='auto',
                   cmap='RdBu_r')
    ax.set_title(ch_type.upper())
    ax.set_xlabel('sources')
    ax.set_ylabel('sensors')
    plt.colorbar(im, ax=ax, cmap='RdBu_r')

###############################################################################
# Show sensitivity of each sensor type to dipoles in the source space

grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')

plt.figure()
plt.hist([grad_map.data.ravel(), mag_map.data.ravel(), eeg_map.data.ravel()],
         bins=20, label=['Gradiometers', 'Magnetometers', 'EEG'],
         color=['c', 'b', 'k'])
plt.title('Normal orientation sensitivity')
plt.xlabel('sensitivity')
plt.ylabel('count')
plt.legend()

# Cautious smoothing to see actual dipoles
grad_map.plot(time_label='Gradiometer sensitivity', subjects_dir=subjects_dir,
              clim=dict(lims=[0, 50, 100]))

# Note. The source space uses min-dist and therefore discards most
# superficial dipoles. This is why parts of the gyri are not covered.
