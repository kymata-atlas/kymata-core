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
experimentName    = ['LexproMEG']; 
wordlistFilename  = [rootDataSetPath, experimentName, '/scripts/Stimuli-Lexpro-MEG-Single-col.txt'];
participentIDlist = cellstr([                                           % participants
            
    'meg08_0320'
    'meg08_0323'
    'meg08_0324'
    'meg08_0327'
    'meg08_0348'
    'meg08_0350'
    'meg08_0363'
    'meg08_0366'
    'meg08_0371'
    'meg08_0372'
    'meg08_0377'
    'meg08_0380'
    'meg08_0397'
    'meg08_0400'
    'meg08_0401'
    'meg08_0402'
                                        ]);
                                    
maxnumberofparts            = 4;
                                
                                

% MEG processing variables
pre_stimulus_window         = 200;                              % in milliseconds
length_of_longest_stimuli   = 853;                              % in milliseconds
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