% setup file for neurokymatography scripts

% Add paths
addpath /imaging/local/linux/mne_2.6.0/mne/matlab/toolbox/;
addpath /imaging/at03/NKG_Code/Version4_2/mne_matlab_functions/;


%-------------------
% Set variables
%-------------------

% Root path variables
rootDataSetPath    = ['/imaging/at03/NKG_Data_Sets/'];
rootCodePath       = ['/imaging/at03/NKG/saved_data/source_data/'];
rootCodeOutputPath = ['/imaging/at03/NKG_Code_output/'];
rootFunctionPath   = ['/imaging/at03/NKG_Data_Functions/'];
version = 'Version4_2';


% Input variables

%Verbphrase
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

% % Lexpro
% experimentName    = ['LexproMEG']; 
% wordlistFilename  = [rootDataSetPath, experimentName, '/scripts/Stimuli-Lexpro-MEG-Single-col.txt'];
% participentIDlist = cellstr([                                           % participants
%             
%     'meg08_0320'
%     'meg08_0323'
%     'meg08_0324'
%     'meg08_0327'
%     'meg08_0348'
%     'meg08_0350'
%     'meg08_0363'
%     'meg08_0366'
%     'meg08_0371'
%     'meg08_0372'
%     'meg08_0377'
%     'meg08_0380'
%     'meg08_0397'
%     'meg08_0400'
%     'meg08_0401'
%     'meg08_0402'
%                                         ]);
%                                     
% maxnumberofparts            = 4;

% MEG processing variables
pre_stimulus_window         = 200;                              % in milliseconds
length_of_longest_stimuli   = 1067;                             % in milliseconds
post_stimulus_window        = 800;                              % in milliseconds
temporal_downsampling_rate  = 100;                              % in Hrz

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


%----------------------------------
% Work out how many blocks there are per participant
%----------------------------------

%create hash table containing number of sessions for each participant,
%in-case there are five sessions.

participentSessionHash = java.util.Hashtable;

for i = 1:numel(participentIDlist)
    eventfilename = [rootDataSetPath, experimentName, '/', char(participentIDlist(i)), '/', char(participentIDlist(i)), '_part' num2str(maxnumberofparts) '-acceptedwordevents.eve'];
    if(exist(eventfilename, 'file'))
        participentSessionHash.put(char(participentIDlist(i)),maxnumberofparts);
    else
        participentSessionHash.put(char(participentIDlist(i)),maxnumberofparts-1);
    end
end
