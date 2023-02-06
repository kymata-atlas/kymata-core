% script for extracting sensor data from preprocessed
% raw .fif files, saving them as .mats, applying the
% single trial inverse matrix, average them, and then
% do statistics on the sources space averages.

% type mne_setup_2.6.0 before strating matlab!
% A

% One-offs
NKG_make_morph_maps;

% prepre-process raw fif files
NKG_create_item_list.pl;
create x4 bad channel .txt files
NKG_mark_bad_channels.sh
NKG_mne_check_eeg_locations.sh
NKG_do_sss_movecomp_tr.sh;
NKG_mark_EEG_bad_channels.sh

% pre-process raw fif files
NKG_remove_eyeblinks;
NKG_make_trials.py;
NKG_getSingleTrialEvokedSourceData;
NKG_make_averaged_meshes;

% apply neurokymatogragphy to averaged data, save p-values in 'saved_data/source_space/4-neurokymatogaphy-data'
NKG_do_neurokymatography;

% plot p-values over time or save for tabula
NKG_plot_pvalues_over_time;
