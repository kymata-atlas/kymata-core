%---------------------------------------------
% This file is a pipe-line that takes a fif. file, triggers and
% model-signals and output the coherence between the MEG and this signal.
%---------------------------------------------

addpath /opt/neuromag/meg_pd_1.2/
addpath /imaging/at03/Fieldtrip/
addpath /opt/mne/matlab/toolbox/
addpath /imaging/at03/Fieldtrip/fileio/


%---------------------------------------------
% Define variables
%---------------------------------------------

% specify filenames
participentIDlist = [19 20 23 24 27];

% Options
prewindow  = 0.2;
postwindow = 0.1;

% global variables
stimuli = [];


%---------------------------------------------
% Import Fif files (max filtered already, downsampling of 45? 50Hz?)
%---------------------------------------------

MEGParticipantArray = [];

for m = 1:length(participentIDlist)

    % create an empty data structure

    MEGtrialdata = [];
    MEGevents = [];

    for q = 1:4  %WHAT ABOUT 4 IN THE LAST ONE???


        MEGfilename = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(q), '_raw_sss_movecomp_tr.fif'];
        eventlistFilename = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(q), '-xxxxxxxxxxx.eve'];

        cfg = [];
        cfg.dataset         = MEGfilename;
        cfg.hearderfile     = MEGfilename;
        cfg.continuous      = 'yes';

        cfg.trialdef.pre    = prewindow;
        cfg.trialdef.post   = postwindow;

        %---------------------------------------------
        % Merge the parts
        %---------------------------------------------
        
        append (data, xxx, xxx, xxx)
        
        %---------------------------------------------
        % Merge the event files
        %---------------------------------------------

        parts
        % import amended eventfile
        fid = fopen(eventlistFilename);
            stimuliIDfile = textscan(fid, '%s %s %s %s %s %s %s %s');
        fclose(fid);

    end

    for word in event;

        make trial Nx3 matix
        trialfun(word);
        cfg.events = [just that xxx];

        %---------------------------------------------
        % Create pre-processed data
        %---------------------------------------------

        % lowpass, and baseline corrected, no artifact rejection as it has
        % been done manually

        cfg.channel    = {'MEG'};                            % read all MEG channels
        cfg.blc        = 'yes';
        cfg.blcwindow  = [-0.2 0];
        cfg.lpfilter   = 'yes';                              % apply lowpass filter
        cfg.lpfreq     = 55;                                 % in Hz (Should be 55)
        cfg.padding    = 0.5;                                % length to which the trials are padded for filtering

        thiswordpreproccessed = preprocessing(cfg);
        
        cfg = [];
        avgAverage AVEword = timelockanalysis(cfg, thiswordpreproccessed);

        save AVErageword ../participant/AVEword

        clear xxx
    end
    clear xxx
end

%---------------------------------------------
% Create grand average
%---------------------------------------------

for word in event
    for each participant
        get word;
            
        % megrealign
        [interp] = megrealign(cfg, data);   %
        % Required configuration options
        cfg.template;                       %
        cfg.inwardshift;                    %
    end
        
    grandaverageword = (dataLodge12, dataDoge13, dataDoge14, dataDoge15, dataDoge16)
    
    save AVErageword ../GrandAverage/G_AVE_word   
end

%---------------------------------------------
% Compress to a single structure
%---------------------------------------------

%Compress to single structure

%---------------------------------------------
% Construct planar gradients (analogous, but not the same as RMRS)
%---------------------------------------------

%planar gradients (GRADS?)


    