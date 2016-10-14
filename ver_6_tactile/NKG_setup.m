% setup file for neurokymatography scripts

% Add paths
addpath /imaging/local/linux/mne_2.6.0/mne/matlab/toolbox/;
addpath /imaging/at03/NKG_Code/Version5/mne_matlab_functions/;


%-------------------
% Set variables
%-------------------

% Root path variables
rootDataSetPath    = ['/imaging/at03/NKG_Data_Sets/'];
rootCodePath       = ['/imaging/at03/NKG/saved_data/source_data/'];
rootCodeOutputPath = ['/imaging/at03/NKG_Code_output/'];
rootFunctionPath   = ['/imaging/at03/NKG_Data_Functions/'];
version = 'Version5';


% Input variables

experimentName    = ['DATASET_3-02_tactile_toes']; 
itemlistFilename  = [rootDataSetPath, experimentName, '/items.txt'];
participentIDlist = cellstr([                                           % participants
                         'meg15_0537_1'
                         'meg15_0537_2'
                                        ]);
                                    
                                    
participentNumBlockHash = java.util.Hashtable;

maxnumberofparts            = 4;

for i = 1:numel(participentIDlist)
    eventfilename = [rootDataSetPath, experimentName, '/', char(participentIDlist(i)), '/block' num2str(maxnumberofparts) '_raw.fif'];
    if(exist(eventfilename, 'file'))
        disp([eventfilename, ', count:2'])
        participentNumBlockHash.put(char(participentIDlist(i)),maxnumberofparts);
    else
        disp([eventfilename, ',count:1'])
        participentNumBlockHash.put(char(participentIDlist(i)),maxnumberofparts-1);
    end
end
                                   

% MEG processing variables
pre_stimulus_window         = 200;                              % in milliseconds
length_of_longest_stimuli   = 1000;                             % in milliseconds
post_stimulus_window        = 800;                              % in milliseconds
temporal_downsampling_rate  = 100;                              % in Hrz

% neurokymatogragphy variables
windowsize                  = 7;                                % diameter


% output variables
latency_step                = 5;                                % in milliseconds




%----------------------------------
% create itemlist
%----------------------------------


fid = fopen(itemlistFilename);
itemlist = textscan(fid, '%s');
itemlist = itemlist{1};
fclose('all');

