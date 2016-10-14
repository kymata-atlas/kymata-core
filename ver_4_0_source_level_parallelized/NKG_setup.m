% setup file for neurokymatography scripts

% Add paths
addpath /imaging/local/linux/mne_2.6.0/mne/matlab/toolbox/;


%-------------------
% Set variables
%-------------------

% Root path variables
rootDataSetPath    = ['/imaging/at03/NKG_Data_Sets/'];
rootCodePath       = ['/imaging/at03/NKG/saved_data/source_data/'];
rootCodeOutputPath = ['/imaging/at03/NKG_Code_output/'];
rootFunctionPath   = ['/imaging/at03/NKG_Data_Functions/'];


% Input variables
experimentName    = ['VerbphraseMEG']; 
wordlistFilename  = [rootDataSetPath, experimentName, '/scripts/Stimuli-Verbphrase-MEG-Single-col.txt'];
participentIDlist = cellstr([                                           % participants
                        'meg10_0003'
                        'meg10_0006'
                        'meg10_0007'
                        'meg10_0009'
                        'meg10_0011'
                        'meg10_0013'
                        'meg10_0019'
                        'meg10_0020'
                        'meg10_0021'
                        'meg10_0022'
                        'meg10_0028'
                        'meg10_0039'
                        'meg10_0040'
                        'meg10_0041'
                        'meg10_0043'
                        'meg10_0045'
                        'meg10_0061'
                        'meg10_0063'
                        'meg10_0073'
                        'meg10_0075'
                                        ]);
                                    
maxnumberofparts            = 5;

% MEG processing variables
pre_stimulus_window         = 200;                              % in milliseconds
length_of_longest_stimuli   = 1067;                             % in milliseconds
post_stimulus_window        = 800;                              % in milliseconds
temporal_downsampling_rate  = 100;                              % in Hrz

% Source space variables
spacial_downsampling        = 1000;                             % vertices per hemisphere
SNR                         = (1/3);                            % third is normal

% neurokymatogragphy variables
windowsize                  = 7;                                % diameter


% output variables
latency_step                = 5;                                % in milliseconds




%----------------------------------
% create wordlist
%----------------------------------

%create full wordlist (i.e. all words)

fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
wordlist = wordlist{1};
fclose('all');