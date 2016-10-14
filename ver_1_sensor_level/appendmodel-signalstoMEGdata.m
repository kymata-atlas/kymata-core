%--------------------------
% Append model-signals to MEG data
%--------------------------


addpath /opt/neuromag/meg_pd_1.2/
addpath /imaging/at03/Fieldtrip/
addpath /opt/mne/matlab/toolbox/
addpath /imaging/at03/Fieldtrip/fileio/
 
 
%--------------------------
% Import and append signals
%--------------------------

load /imaging/at03/Fieldtrip_recogniser_coherence/saved_data/MEGpreprocessed_participant_080319FULL-corr.mat;

window = 3;

% Align everything

%cfg = [];
%timelockedMEG = timelocked(cfg,meg_paart19);

%cfg.template    =  4;
%cfg.inwardshift =  1;
%[intep] = megrealign(cfg, timelockedMEG);


% Import stimuli order and signals for each participant

load /imaging/at03/Fieldtrip_recogniser_coherence/saved_data/stimuli.mat;
load /imaging/at03/Fieldtrip_recogniser_coherence/signals/C0/stimulisig.mat;

% deleate trials that haven't been recognised

stimulisigfilenames = '/imaging/at03/Fieldtrip_recogniser_coherence/signals/C0/aa_names.txt';

names = importdata(stimulisigfilenames);

for p = 1:length([19])
    numberoftrials = length(meg_part19.trial);
    keepmatrix = zeros(1,numberoftrials); % must be changed for more than one person
    for n = 1:length(names)
        issame = strcmp(names{n,1}, stimuli.values{1,1}());
        keepmatrix = keepmatrix | issame(1, 1:numberoftrials);
    end
    for i = numberoftrials:-1:1   %for each of the trials delete the correspionding empties
        if(keepmatrix(1,i) == 0)
            meg_part19.trial(:,i) = [];
            meg_part19.time(:,i)  = [];
            stimuli.values{1,1}(:,i) = [];
        end
    end
end


% Average all trials that are the same, from all participants

%for each word,
%    look in each participant
%    find the chorrect trial(s)
%    place this trial in a new structure
%    append a stimuli name to it
%end

% Attatch true entropy dataset (lable: 'SIGtrue')

for p=1:length([19])
    for i = 1:length(meg_part19.trial)
        %Find word in stimuli for participant 1
        thisstimuliword = stimuli.values{1,1}(1,i);
        meg_part19.label{307} = 'SIGtrue';
        thisstimuliwordposition = find (strcmp(thisstimuliword, stimulisig.name));
        for j=1:size(stimulisig.time, 2);
            if (stimulisig.time(thisstimuliwordposition, j) ~= 0)
                thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
                thissig = stimulisig.C0(thisstimuliwordposition, j); %accoustic likelhood signal
                thistimeposinMEG = find (meg_part19.time{1,i} == thistime);
                meg_part19.trial{1,i}(307, thistimeposinMEG) = thissig;
            end
        end
    end
end

% Attatch jumbled entropy dataset (lable: 'SIGrandom')

randomperm = randperm(length(meg_part19.trial));

for p=1:length([19])
    for i = 1:length(meg_part19.trial)
        %Find word in stimuli for participant 1
        thisstimuliword = stimuli.values{1,1}(1,randomperm(i));
        meg_part19.label{308} = 'SIGrand';
        thisstimuliwordposition = find (strcmp(thisstimuliword, stimulisig.name));
        for j=1:size(stimulisig.time, 2);
            if (stimulisig.time(thisstimuliwordposition, j) ~= 0)
                thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
                thissig = stimulisig.C0(thisstimuliwordposition, j);
                thistimeposinMEG = find (meg_part19.time{1,i} == thistime);
                meg_part19.trial{1,i}(308, thistimeposinMEG) = thissig;
            end
        end
    end
end


% Replace zeros with NaNs
      
for i = 1:length(meg_part19.trial) % for each word
    disp(num2str(i));
    for j=1:length(meg_part19.trial{1,i})
            disp([num2str(i), ':', num2str(j)]);
            meg_part19.trial{1,i}(meg_part19.trial{1,i}==0)=NaN;
    end
end