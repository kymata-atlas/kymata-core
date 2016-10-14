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

clear all
participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];

load /imaging/at03/Fieldtrip_recogniser_coherence/signals/C0/stimulisig.mat;
wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');

inputsarray = {};
for thiswordpos = 1:length(wordlist{1,1})
    inputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/4-grandaveraged/', wordlist{1,1}{thiswordpos,1}, '.mat'];
    if (exist(inputfilename, 'file'))
        load (inputfilename);
        inputsarray{i,1} = wordlist{1,1}{i,1};
    end
end

window = 3;

% Attatch true entropy dataset (lable: 'SIGtrue')

for thiswordpos = 1:length(inputsarray)

       %Find word in stimuli for participant 1
       thisstimuliword = inputsarray{thiswordpos,1};
eval([ 'grandaverage_', thisstimuliword, '.label{307} = ''', 'SIGtrue', ''';' ])
       thisstimuliwordposition = find (strcmp(thisstimuliword, stimulisig.name));
       for j=1:size(stimulisig.time, 2);
           if (stimulisig.time(thisstimuliwordposition, j) ~= 0)
               thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
               thissig = stimulisig.C0(thisstimuliwordposition, j); %accoustic likelhood signal
               thistimeposinMEG = find (meg_part19.time{1,i} == thistime);
eval([        'grandaverage_', thisstimuliword, '.trial{1,i}(307, thistimeposinMEG) = thissig;' ]);
           end
        end
end

% Attatch jumbled entropy dataset (lable: 'SIGrandom')


randomperm = randperm(length(meg_part19.trial));


for i = 1:length(inputsarray)
        %Find word in stimuli for participant 1
        thisstimuliword = inputsarray{randomperm(i),1};
eval([ 'grandaverage_', thisstimuliword, '.label{308} = ''', 'SIGrand', ''';' ]);
        thisstimuliwordposition = find (strcmp(thisstimuliword, stimulisig.name));
        for j=1:size(stimulisig.time, 2);
            if (stimulisig.time(thisstimuliwordposition, j) ~= 0)
                thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
                thissig = stimulisig.C0(thisstimuliwordposition, j);
                thistimeposinMEG = find (meg_part19.time{1,i} == thistime);
eval([          'grandaverage_', thisstimuliword, '.trial{1,i}(308, thistimeposinMEG) = thissig;' ]);
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