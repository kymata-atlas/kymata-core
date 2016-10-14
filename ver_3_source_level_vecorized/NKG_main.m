% script for extracting sensor data from preprocessed
% raw .fif files, saving them as .mats, applying the
% single trial inverse matrix, average them, and then
% do statistics on the sources space averages.

% type mne_setup_2.6.0 before strating matlab!
% A

% Add paths
addpath /imaging/local/linux/mne_2.6.0/mne/matlab/toolbox/

% Global variables
process                     = 'C0';                         % name of variable
processfilename             = 'C0_deltaC0_deltadeltaC0';    % name of the file the process is in
length_of_longest_stimuli   = 852;                          % ????????????????? %in milliseconds
latency_step                = 10;                           % in milliseconds
pre_stimulus_window         = 100;                          % in milliseconds
post_stimulus_window        = 400;                          % in milliseconds
temporal_downsampling_rate  = 55;                           % in hz  

% MEG processing variables


% Source space variables
spacial_downsampling        = 1000;                         % vertices per hemisphere
SNR                         = (1/3);                        % third is normal

% neurokymatogragphy variables
windowsize                  = 7;                            % diameter



% One-offs
%NKG_make_morph_maps;
%NKG_make_emptynoise_and_inverse;

% pre-process raw fif files and save them in 'saved_data/source_space/1-sensor-data'
NKG_setup;
NKG_generateSingleTrialEventfiles;
NKG_getSingleTrialEvokedSensorDataWithMNE;
NKG_resortAllsingleTrialsOutOfsessions;
NKG_make_averaged_repetitions;
% apply single trial inverse matrix, then obatin source_data and save them in 'saved_data/source_space/2-single-trial-data'
NKG_getSingleTrialEvokedSourceData;
% average the single trail data and save them in
% 'saved_data/source_space/3-averaged-by-trial-data'
NKG_make_averaged_meshes;
% apply neurokymatogragphy to averaged data, save p-values in 'saved_data/source_space/4-neurokymatogaphy-data'
NKG_do_neurokymatography_lh;
NKG_do_neurokymatography_rh;
% display neurokymatogaphy p-value data
NKG_display_neurokymatography;