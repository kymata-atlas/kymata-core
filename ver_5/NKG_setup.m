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

experimentName    = ['DATASET_3-01_visual-and-auditory']; 
itemlistFilename  = [rootDataSetPath, experimentName, '/items.txt'];
participentIDlist = cellstr([                                           % participants
                         'meg15_0045'
                         'meg15_0051'
                         'meg15_0054'
                         'meg15_0055'
                         'meg15_0056'
                         'meg15_0058'
                         'meg15_0060'
                         'meg15_0065'
                         'meg15_0066'
                         'meg15_0070'
                         'meg15_0071'
                         'meg15_0072'
                         %'meg15_0079'
                         'meg15_0081'
                         'meg15_0082'
                                        ]);
                                    
                                    
participentNumBlockHash = java.util.Hashtable;

maxnumberofparts            = 2;

for i = 1:numel(participentIDlist)
    eventfilename = [rootDataSetPath, experimentName, '/', char(participentIDlist(i)), '/nkg_part' num2str(maxnumberofparts) '_raw.fif'];
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

% stimulus delivery latency
audio_delivery_latency	    = 16;                               % in milliseconds
visual_delivery_latency	    = 34;                              % in milliseconds

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

