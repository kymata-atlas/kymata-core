%--------------------------
% Find corrolation
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

% -------------------------------
% Find correlation co-effients for both
% -------------------------------

truewords = [];   
randwords = [];

for i = 1:length(meg_part19.trial) % for each word
    
    %-------------------
    % for the trueSig
    %-------------------
    trueSig = meg_part19.trial{i}(307,:);
    % get rid of NaNs
    keepSigtime = meg_part19.time{i}((~isnan(meg_part19.time{i})) & (~isnan(trueSig)));
    keeptrueSig = trueSig((~isnan(meg_part19.time{i})) & (~isnan(trueSig)));


    for j = 1:306
        disp([num2str(i), ':', num2str(j)]);
        averagechannelsig = [];
        for k = 1:length(keeptrueSig)
            channeltimepos = find (meg_part19.time{i}(1,:) == keepSigtime(k));
            average = mean(meg_part19.trial{i}(j,(channeltimepos-window):(channeltimepos+window)));
            averagechannelsig(k) = average;
        end
        
        % Plot it
        %scatter(averagechannelsig,keeptrueSig, 3), xlabel('averagechannelsig'), ylabel('keeptrueSig');
        
        % Record correlation
        truewords(j,i) = corr2(averagechannelsig, keeptrueSig);

        %[R,P]=corrcoef(averagechannelsig, keeptrueSig)
        % cov(averagechannelsig, keeptrueSig)
        % Non-parametric correlation coefficients
        % Multivariate normal distribution
        % cov(averagechannelsig, keeptrueSig)

    end
end

for i = 1:length(meg_part19.trial) % for each word

    %-------------------
    % for the randSig
    %-------------------
    randSig = meg_part19.trial{i}(308,:);
    % get rid of NaNs
    keepSigtime = meg_part19.time{i}((~isnan(meg_part19.time{i})) & (~isnan(randSig)));
    keeprandSig = randSig((~isnan(meg_part19.time{i})) & (~isnan(randSig)));

    for j = 1:306
        disp([num2str(i), ':', num2str(j)]);
        averagechannelsig = [];
        for k = 1:length(keeprandSig)
            channeltimepos = find (meg_part19.time{i}(1,:) == keepSigtime(k));
            average = mean(meg_part19.trial{i}(j,(channeltimepos-window):(channeltimepos+window)));
            averagechannelsig(k) = average;
        end
        
        %Plot it
        scatter(averagechannelsig,keeprandSig, 3), xlabel('averagechannelsig'), ylabel('keeprandSig');
        
        % Record correlation
        randwords(j,i) = corr2(averagechannelsig, keeprandSig);

        %[R,P]=corrcoef(averagechannelsig, keeptrueSig)
        % cov(averagechannelsig, keeptrueSig)
        % Non-parametric correlation coefficients
        % Multivariate normal distribution
        % cov(averagechannelsig, keeptrueSig)
    end
end
%-------------------------------
% Do population test on signals
%-------------------------------

%make discete
pvalues = []; %for elizabeth
sigvalue = 0.005;
display(['The null hypothesis is that the specified channel does does not significantly encode TRUESIG information any more than it encodes for RANSIG.']);
display(['The significance value is ', num2str(sigvalue) ]);
for channel = 1:306
    truepopulation = truewords(channel, :);
    randpopulation  = randwords(channel, :);
    [h,p,ci,zval] = ztest(truepopulation, mean(randpopulation), std(randpopulation));
    if(p<sigvalue)
        display(['Channel ', num2str(channel), '. Null hypothesis for this channel rejected at p-value ', num2str(p) ,'*']);
    else
        display(['Channel ', num2str(channel), '. Null hypothesis not rejected (p value:', num2str(p) , ')']);
    end
    pvalues = [pvalues p]; %for elizabeth
    %output.xy = [output.xy{1,2}; getxyfromlayout()];
end