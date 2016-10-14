
addpath /opt/neuromag/meg_pd_1.2/
addpath /imaging/at03/Fieldtrip/
addpath /opt/mne/matlab/toolbox/
addpath /imaging/at03/Fieldtrip/fileio/


%------------------
% Display coherence
%------------------

%load /imaging/at03/Fieldtrip_recogniser_coherence/saved_data/MEGpreprocessed_participant_080319FULL-corr.mat;

% Align everything

cfg = [];
timelockedMEG = timelocked(cfg,meg_paart19);

cfg.template    =  4;
cfg.inwardshift =  1;
[intep] = megrealign(cfg, timelockedMEG);

% Import entropy for each participant

%load /imaging/at03/Fieldtrip_recogniser_coherence/saved_data/stimuli.mat;





% Attatch true entropy dataset (lable: 'SIGtrue')

basepoint=0;

for i=1:length(meg_part19.trial)
    %Find word in stimuli for participant 1
    thisstimuliword = stimuli.values(1,i);
    load /imaging/at03/Fieldtrip_recogniser_coherence/saved_data/Entropytestsignals.mat
    meg_part19.label(307) = 'SIGtrue';
    for j=basepoont:1
        meg_part19.label(307, j) = sigmatrix;
    end
end

% Attatch jumbled entropy dataset (lable: 'SIGrandom')

randomperm = randperm(1:length(meg_part19.trials));

for i=1:length(meg_part19.trials)
    %Find word in stimuli for participant 1
    thisstimuliword = stimuli(1,i);
    load /imaging/at03/Fieldtrip_recogniser_coherence/saved_data/Entropytestsignals.mat
    meg_part19.lable(307) = 'SIGran';
    for j=basepoint:1
        appeand correct word from participant 19 as 'SIGran';
    end
end

%find coherence that does best with itself (in order to make sure we have the right one)

%------------------
% Find coherence
%------------------

% coherence analyisis (true)
cfg = [];
cfg.output      = 'powandcsd';
cfg.method      = 'mtmfft';
cfg.taper       = 'hanning';
cfg.foilim      = [0 50];
cfg.channel     = {'MEG'};
cfg.channelcmb  = {'MEG' 'SIGtrue'};
cfg.trials = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]

freqData        = freqanalysis(cfg, meg_part19);
freqDescData    = freqdescriptives([],freqData);

% coherence analyisis (random)
cfg = [];
cfg.output      = 'powandcsd';
cfg.method      = 'mtmfft';
cfg.taper       = 'hanning';
cfg.foilim      = [0 50];
cfg.channel     = {'MEG'};
cfg.channelcmb  = {'MEG' 'SIGran'};
cfg.trials = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]

freqDataRAN        = freqanalysis(cfg, meg_part19);
freqDescDataRAN    = freqdescriptives([],freqDataRAN);

%plot

cfg                  = [];
cfg.layout           = 'CBU_NM306mag.lay';
cfg.showlabels       = 'yes';
cfg.interactive      = 'yes';
cfg.zparam           = 'cohspctrm';
cfg.xlim             = [2 50];
cfg.ylim             = [0 1];
cfg.cohrefchannel    = {'SIGran'};
multiplotER(cfg,freqDescData, freqDescDataRAN);






























%------------------
% Find best coherence
%------------------

% coherence analyisis (true)
cfg = [];
cfg.output      = 'powandcsd';
cfg.method      = 'mtmfft';
cfg.taper       = 'hanning';
cfg.foilim      = [0 50];
cfg.channel     = {'MEG'};
cfg.channelcmb  = {'MEG' 'MEG2531'};
cfg.trials = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]

freqData        = freqanalysis(cfg, meg_part19);
freqDescData    = freqdescriptives([],freqData);

% coherence analyisis (random)
cfg = [];
cfg.output      = 'powandcsd';
cfg.method      = 'mtmfft';
cfg.taper       = 'hanning';
cfg.foilim      = [0 50];
cfg.channel     = {'MEG'};
cfg.channelcmb  = {'MEG' 'MEG1941'};
cfg.trials = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]

freqDataRAN        = freqanalysis(cfg, meg_part19);
freqDescDataRAN    = freqdescriptives([],freqDataRAN);

%plot

cfg                  = [];
cfg.layout           = 'CBU_NM306mag.lay';
cfg.showlabels       = 'yes';
cfg.interactive      = 'yes';
cfg.zparam           = 'cohspctrm';
cfg.xlim             = [2 50];
cfg.ylim             = [0 1];
cfg.cohrefchannel    = {'MEG1941'};
multiplotER(cfg,freqDescDataRAN);