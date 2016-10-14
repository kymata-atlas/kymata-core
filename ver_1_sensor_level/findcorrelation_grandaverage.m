%--------------------------
% Find corrolation
%--------------------------


addpath /opt/neuromag/meg_pd_1.2/
addpath /imaging/at03/Fieldtrip/
addpath /opt/mne/matlab/toolbox/
addpath /imaging/at03/Fieldtrip/fileio/
addpath /imaging/at03/Method_to_locate_neurocorrelates_of_processes/

% set variables

clear all

sensortype = 'RMS_then_GRANDAVE';       % 'GRANDAVE_then_MAGS' or 'GRANDAVE_then_RMS'or 'RMS_then_GRANDAVE'
window = 7;                                % miliseconds
epsilon = 0.00000001;


 % do stuff with variables
 
 if (strcmp(sensortype, 'GRANDAVE_then_MAGS'))
     inputfolder = '5-grandaveragenovar';
 elseif (strcmp(sensortype, 'RMS_then_GRANDAVE'))
     inputfolder = 'RMSthenGA/6-RMSgrandaveragenovar';
 else
     inputfolder = '6-grandaveragenovarRMS';
 end

%--------------------------
% Import and append signals
%--------------------------

participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];

wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');
 
inputsarray{400,1} = 0;
for i = 1:length(wordlist{1,1})
    inputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/lpfilter_30Hz/', inputfolder, '/', wordlist{1,1}{i,1}, '.mat'];
    disp(num2str(i));
    if (exist(inputfilename, 'file'))
        load (inputfilename);
        inputsarray{i,1} = wordlist{1,1}{i,1};
        fclose('all');
    end
end

% Import stimuli order and signals for each participant

load /imaging/at03/Method_to_locate_neurocorrelates_of_processes/signals/C0/stimulisig.mat;

% Attatch true entropy dataset (lable: 'SIGtrue')

for i = 1:length(inputsarray)
        %Find word in stimuli for participant 1
        
        thisstimuliword = inputsarray{i,1};
        eval([ 'MEGWORD = grandaverage_', thisstimuliword ]);
        MEGWORD.label{307} = 'SIGtrue';
        thisstimuliwordposition = find (strcmp(thisstimuliword, stimulisig.name));
        for j=1:size(stimulisig.time, 2);
            if (stimulisig.time(thisstimuliwordposition, j) ~= 0)
                thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
                thissig = stimulisig.C0(thisstimuliwordposition, j); %accoustic likelhood signal
                thistimeposinMEG = find(abs(thistime - MEGWORD.time(1,:)) < epsilon); %floating decimal point problem
                MEGWORD.avg(307, thistimeposinMEG) = thissig;
            end
        end
        eval([ 'grandaverage_', thisstimuliword, ' = MEGWORD' ]);
end

%generate random array, so that the baselines are never true 
randomperm = zeros(50,length(inputsarray));
for p=1:50
    bagofnumbers = randperm(length(inputsarray));
    if (bagofnumbers(length(bagofnumbers)) == length(bagofnumbers))
        bagofnumbers(length(bagofnumbers)) = bagofnumbers(1);
        bagofnumbers(1) = length(bagofnumbers);
    end
    for l=1:length(bagofnumbers)
        if (bagofnumbers(l) == l)
            temp = bagofnumbers(l+1);
            bagofnumbers(l+1) = bagofnumbers(l);
            bagofnumbers(l) = temp;
        end
        randomperm(p,l) = bagofnumbers(l);
    end
end

%Attatch jumbled entropy dataset (lable: 'SIGrandom')

for i = 1:length(inputsarray)
        thisstimuliword = inputsarray{i,1};
        eval([ 'MEGWORD = grandaverage_', thisstimuliword ]);
        for channel = 308:357
            ranpermnumber = channel-307;
            %Find word in stimuli for participant 1
            thisstimuliRANword = inputsarray(randomperm(ranpermnumber, i), 1);
            MEGWORD.label{channel} = ['SIGrand', num2str(ranpermnumber)];
            thisstimuliwordposition = find(strcmp(thisstimuliRANword, stimulisig.name));
            for j=1:size(stimulisig.time, 2);
                if (stimulisig.time(thisstimuliwordposition, j) ~= 0)
                    thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
                    thissig = stimulisig.C0(thisstimuliwordposition, j);
                    thistimeposinMEG = find(abs(thistime - MEGWORD.time(1,:)) < epsilon); %floating decimal point problem
                    MEGWORD.avg(channel, thistimeposinMEG) = thissig;
                end
            end
        end
        eval([ 'grandaverage_', thisstimuliword, ' = MEGWORD;' ]);
end



% Replace zeros with NaNs
      
for i = 1:length(inputsarray) % for each word
    disp(num2str(i));
    thisstimuliword = inputsarray{i,1};
    eval([ 'MEGWORD = grandaverage_', thisstimuliword, ';' ]);
    for j=1:length(MEGWORD.time)
            disp([num2str(i), ':', num2str(j)]);
            MEGWORD.avg(MEGWORD.avg==0)=NaN;
    end
    eval([ 'grandaverage_', thisstimuliword, ' = MEGWORD' ]);
end

% -------------------------------
% Find correlation co-effients for both
% -------------------------------

truewords = zeros(306,length(inputsarray));
randwords = zeros(306,(length(inputsarray)*50));

for i = 1:length(inputsarray) % for each word

     thisstimuliword = inputsarray{i,1};
     eval([ 'MEGWORD = grandaverage_', thisstimuliword, ';' ]);
     
    
    %-------------------
    % for the trueSig
    %-------------------
    trueSig = MEGWORD.avg(307,:);
    % get rid of NaNs
    keepSigtime = MEGWORD.time((~isnan(MEGWORD.time(i))) & (~isnan(trueSig)));
    keeptrueSig = trueSig((~isnan(MEGWORD.time(i))) & (~isnan(trueSig)));


    for j = 1:306
        disp([num2str(i), ':', num2str(j)]);
        averagechannelsig = [];
        for k = 1:length(keeptrueSig)
            channeltimepos = find (MEGWORD.time(1,:) == keepSigtime(k));
            % average = mean(hamming(window)' .* MEGWORD.avg(j,(channeltimepos-(ceil(window/2))+1):(channeltimepos-1+(ceil(window/2)))));
            average = mean(MEGWORD.avg(j,(channeltimepos-(ceil(window/2))+1):(channeltimepos-1+(ceil(window/2)))));
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
        
        thisstimuliword = inputsarray{i,1};
        eval([ 'MEGWORD = grandaverage_', thisstimuliword, ';' ]);

        
        %-------------------
        % for the randSig
        %-------------------
        randSig = MEGWORD.avg(permnumber,:);
        % get rid of NaNs
        keepSigtime = MEGWORD.time((~isnan(MEGWORD.time(i))) & (~isnan(randSig)));
        keeprandSig = randSig((~isnan(MEGWORD.time(i))) & (~isnan(randSig)));
        keeprandsigcopy  = keeprandSig;
        
        for j = 1:306
            keeprandSig = keeprandsigcopy; % deleat as soon as you get the chance....
            disp([num2str(i), ':', num2str(j), ':', num2str(permnumber)]);
            averagechannelsig = [];
            for k = 1:length(keeprandSig)
                channeltimepos = find(MEGWORD.time(1,:) == keepSigtime(k));
                if (channeltimepos+window > length(MEGWORD.time))
                    keeprandSig(k:end)=[];
                    
                    break
                else
                    %average = mean(hamming(window)' .* MEGWORD.avg(j,(channeltimepos-(ceil(window/2))+1):(channeltimepos-1+(ceil(window/2)))));
                    average = mean(MEGWORD.avg(j,(channeltimepos-(ceil(window/2))+1):(channeltimepos-1+(ceil(window/2)))));
                    averagechannelsig(k) = average;
                end
            end
            

            % Plot it
            % scatter(averagechannelsig,keeprandSig, 3), xlabel('averagechannelsig'), ylabel('keeprandSig');

            % Record correlation: pearson's Rho
            index = i + (length(inputsarray) * (permnumber-308));
            randwords(j,index) = corr2(averagechannelsig, keeprandSig);

        end
    end
end

%-------------------------------
% Distribution tests
%-------------------------------

%transform using Fisher's Z

randwordsGuassian = zeros(size(randwords, 1), size(randwords, 2));
truewordsGuassian = zeros(size(truewords, 1), size(truewords, 2));

for word = 1:size(randwords, 2)
  for channel = 1:306
      r = randwords(channel,word);
      randwordsGuassian(channel,word) = 0.5*log((1+r)/(1-r));
  end
end

for word = 1:size(truewords, 2)
  for channel = 1:306
      r = truewords(channel,word);
      truewordsGuassian(channel,word) = 0.5*log((1+r)/(1-r));
  end
end

%lillefor test to check for Guassian

h = lillietest(randwordsGuassian(36,:))

histfit(randwordsGuassian(132,:), 30); % plot
hold on; 
histfit(truewordsGuassian(132,:), 30); 
 
h = findobj(gca,'Type', 'patch');
display(h);
  
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
    truepopulation = truewordsGuassian(channel, :);
    randpopulation  = randwordsGuassian(channel, :);
    %[p,h,stats] = ranksum(truepopulation, randpopulation);  %
    %non-parametric
    [h,p,ci,zval] = ztest(truepopulation,mean(randpopulation),std(randpopulation));
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
cfg.ecolor          = [1 1 1];              % Marker color (default = [0 0 0] (black))
cfg.highlight       = highlightchannels;    % or the channel numbers you want to highlight (default = 'off'). These numbers should correspond with the channels in the data, not in the layout file.
cfg.hlcolor         = [1 1 1];

topoplot(cfg, datavector)

