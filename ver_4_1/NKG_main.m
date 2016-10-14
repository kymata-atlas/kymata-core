% script for extracting sensor data from preprocessed
% raw .fif files, saving them as .mats, applying the
% single trial inverse matrix, average them, and then
% do statistics on the sources space averages.

% type mne_setup_2.6.0 before strating matlab!
% A

% One-offs
NKG_make_morph_maps;
%NKG_make_emptynoise_and_inverse;

% pre-process raw fif files
NKG_setup;
NKG_generateSingleTrialEventfiles;
NKG_getSingleTrialEvokedSensorDataWithMNE;
NKG_getSingleTrialEvokedSourceData;
NKG_make_averaged_meshes;

% apply neurokymatogragphy to averaged data, save p-values in 'saved_data/source_space/4-neurokymatogaphy-data'
NKG_do_neurokymatography;

% display neurokymatogaphy p-value data
NKG_display_neurokymatography;

% plot p-values over time
NKG_plot_pvalues_over_time;