%--------------------------
% Find corrolation
%--------------------------


addpath /opt/neuromag/meg_pd_1.2/
addpath /imaging/at03/Fieldtrip/
addpath /opt/mne/matlab/toolbox/
addpath /imaging/at03/Fieldtrip/fileio/
addpath /imaging/at03/Method_to_locate_neurocorrelates_of_processes/
 
 
%--------------------------
% Import and append signals
%--------------------------

% clear all
% participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];
% 
% load /imaging/at03/Method_to_locate_neurocorrelates_of_processes/signals/C0/stimulisig.mat;
% wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
% fid = fopen(wordlistFilename);
% wordlist = textscan(fid, '%s');
% fclose('all');
% 
% inputsarray = {};
% for i = 1:length(wordlist{1,1})
%     disp(num2str(i));
%     inputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/4-grandaverage/', wordlist{1,1}{i,1}, '.mat'];
%     if (exist(inputfilename, 'file'))
%         %load (inputfilename);
%         inputsarray{i,1} = wordlist{1,1}{i,1};
% eval([  'clear grandaverage_', wordlist{1,1}{i,1} ]);
%     end
% end

window = 3;

% Import stimuli order and signals for each participant

load /imaging/at03/Method_to_locate_neurocorrelates_of_processes/signals/C0/stimulisig.mat;

% Attatch true entropy dataset (lable: 'SIGtrue')

randomperm = randperm(length(inputsarray));

for i = 1:length(inputsarray)

    
       %Find word in stimuli for participant 1
       thisstimuliword = inputsarray{i,1};
       inputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/4-grandaverage/', thisstimuliword, '.mat'];
       load (inputfilename);
eval([ 'MEGWORD = grandaverage_', thisstimuliword ]);
eval([ 'clear grandaverage_', thisstimuliword ]);
       MEGWORD.label{307} = 'SIGtrue';
       thisstimuliwordposition = find (strcmp(thisstimuliword, stimulisig.name));
       for j=1:size(stimulisig.time, 2);
           if (stimulisig.time(thisstimuliwordposition, j) ~= 0)
               thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
               thissig = stimulisig.C0(thisstimuliwordposition, j); %accoustic likelhood signal
               thistimeposinMEG = find (MEGWORD.time(1,i) == thistime);
               MEGWORD.ave{1,i}(307, thistimeposinMEG) = thissig;
           end
       end



% Attatch jumbled entropy dataset (lable: 'SIGrandom')


        %Find word in stimuli for participant 1
        for channel = 308:357
            ranpermnumber = channel-307;
            %Find word in stimuli for participant 1
            thisRANstimuliword = stimuli.values{1,1}(1,randomperm(ranpermnumber, i));
            if (thisstimuliword eq thisRANstimuliword)
            MEGWORD.label{channel} = ['SIGrand', num2str(ranpermnumber)];
            thisstimuliwordposition = find (strcmp(thisRANstimuliword, stimulisig.name));
            for j=1:size(stimulisig.time, 2);
                if (stimulisig.time(thisstimuliwordposition, j) ~= 0)
                    thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
                    thissig = stimulisig.C0(thisstimuliwordposition, j);
                    thistimeposinMEG = find (MEGWORD.time(1,i) == thistime);
                    MEGWORD.ave{1,i}(channel, thistimeposinMEG) = thissig;
                end
            end
        end
        




% Replace zeros with NaNs
      

    disp(num2str(i));
    thisstimuliword = inputsarray{thiswordpos,1};
eval([ 'timelength = length(grandaverage_', thisstimuliword, '.trial{1,i})' ]);
    for j=1:timelength;
        disp([num2str(i), ':', num2str(j)]);
eval([ 'grandaverage_', thisstimuliword, '.trial{1,i}(grandaverage_', thisstimuliword, '.trial{1,i}==0)=NaN;' ]);
    end
    GAVEword = ['grandaverage_', wordlist{1,1}{thiswordpos,1}];
    outputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/5-withsig/', wordlist{1,1}{thiswordpos,1}, '.mat'];
    eval([  'save( outputfilename, ''', GAVEword, ''')' ]);

eval([  'clear grandaverage_', wordlist{1,1}{i,1} ]);    
end

%-------------------------------
% Find correlation co-effients for both
%-------------------------------

truewords = [];
randwords = [];

for i = 1:length(inputsarray) % for each word

    thisstimuliword = inputsarray{thiswordpos,1};
    
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
        % scatter(averagechannelsig,keeptrueSig, 3), xlabel('averagechannelsig'), ylabel('keeptrueSig');

        % Record correlation: pearson's Rho ('corr2' tested against the 'corr' version, and it's the same)
        truewords(j,i) = corr2(averagechannelsig, keeptrueSig);

    end
end

for permnumber = 308:357
    for i = 1:length(inputsarray) % for each word
        thisstimuliword = inputsarray{thiswordpos,1};

        %-------------------
        % for the randSig
        %-------------------

        randSig = meg_part19.trial{i}(permnumber,:);
        % get rid of NaNs
        keepSigtime = meg_part19.time{i}((~isnan(meg_part19.time{i})) & (~isnan(randSig)));
        keeprandSig = randSig((~isnan(meg_part19.time{i})) & (~isnan(randSig)));

        for j = 1:306
            disp([num2str(i), ':', num2str(j), ':', num2str(permnumber)]);
            averagechannelsig = [];
            for k = 1:length(keeprandSig)
                channeltimepos = find (meg_part19.time{i}(1,:) == keepSigtime(k));
                average = mean(meg_part19.trial{i}(j,(channeltimepos-window):(channeltimepos+window)));
                averagechannelsig(k) = average;
            end

            % Plot it
            % catter(averagechannelsig,keeprandSig, 3), xlabel('averagechannelsig'), ylabel('keeprandSig');

            % Record correlation: pearson's Rho
            index = i + (length(meg_part19.trial) * (permnumber-308));
            randwords(j,index) = corr2(averagechannelsig, keeprandSig);

        end
    end
end

%-------------------------------
% normalised distribution test
%-------------------------------

[ltesth,ltestp] = lillietest(randwords(3,:))

% plot
histfit(randwords(3,:), 30); 
hold on; 
histfit(truewords(3,:), 30); 
 
h = findobj(gca,'Type', 'patch');
display(h)
 
set(h(1),'FaceColor',[0.982 0.909 0.721],'EdgeColor','k'); 
set(h(2),'FaceColor',[0.819 0.565 0.438],'EdgeColor','k'); 


 
%-------------------------------
% Do population test on signals
%-------------------------------

%make discete
pvalues = [];
sigvalue = 0.05;
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
    pvalues = [pvalues p];
end

%---------------------------------
% Prepare for topoplot
%---------------------------------

MAGpvalues = [];
highlightchannels = [];
i=1;
for channel = 3:3:306
    MAGpvalues =[MAGpvalues ; pvalues(channel)];
    if (pvalues(channel)<sigvalue)
       highlightchannels = [highlightchannels i];
    end
    i=i+1;
end

datavector = MAGpvalues;

%---------------------------------
% Print in topoplot
%---------------------------------

cfg.colormap        = pink;
cfg.layout          = 'CBU_NM306mag.lay';
cfg.colorbar        = 'WestOutside';        % outside left
cfg.gridscale       = 100;                  % scaling grid size (default = 67)
cfg.maplimits       = [0 1];                % Y-scale
cfg.style           = 'both';               %(default)
cfg.contournum      = 9;                    %(default = 6), see CONTOUR
cfg.shading         = 'flat';               %(default = 'flat')
cfg.interpolation   = 'v4';                 % default, see GRIDDATA
cfg.electrodes      = 'highlights';         % should be 'highlights' for white dots. But also 'off','labels','numbers','highlights' or 'dotnum' (default = 'on')
cfg.ecolor          = [0 0 0];              % Marker color (default = [0 0 0] (black))
cfg.highlight       = highlightchannels;    % or the channel numbers you want to highlight (default = 'off'). These numbers should correspond with the channels in the data, not in the layout file.
cfg.hlcolor         = [1 1 1];

topoplot(cfg, datavector)

