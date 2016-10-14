%--------------------------
% Neurokymography on sensors
%--------------------------


% this script works on signals sampled at 1000 times a second, starting at
% t=1. AT


% Parallel===============================================================
%if matlabpool('size') == 0
%    matlabpool open 5
%end


% Paths==================================================================
addpath /imaging/at03/Fieldtrip/
addpath /imaging/at03/NKG/
addpath /imaging/at03/matlab_functions/


% Global variables=======================================================
latencies = -30:5:500;                                                              % what latencies
cutoff = 400;                                                                       % cutt-off for signal times
averagingwindow = 7;                                                                % diameter of window
sigvalues = [0.000005 0.00001 ]; %[0.001 0.0001 0.000001];                           % significance values we are tesing for
nWords = 400;                                                                       % number of words
nTimePoints = 1501;                                                                 % number of time points in raw data
outputpath = '/home/at03/Named_screenshots2/30hzlpfilter_cf400_grandave_then_RMS2/C0/flatwindow7/';    % outputpath for pictures
nSensors = 306;                                                                     % number of sensors


% Initialise=============================================================


% Import raw data 

wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');
 
inputsarray{400,1} = 0;
for i = 1:length(wordlist{1,1})
    inputfilename = ['/imaging/at03/NKG/saved_data/lpfilter_30Hz/5-grandaveragenovar/', wordlist{1,1}{i,1}, '.mat'];
    disp(num2str(i));
    if (exist(inputfilename, 'file'))
        load (inputfilename);
        inputsarray{i,1} = wordlist{1,1}{i,1};
        fclose('all');
    end
end

% import stimuli

load /imaging/at03/NKG/signals/C0_deltaC0_deltadeltaC0/stimulisig.mat;

% create word-signals true

wordsignalstrue = zeros(length(inputsarray), size(stimulisig.C0, 2));

for i = 1:length(inputsarray)
        thisstimuliword = inputsarray{i,1};
        wordsignalstrue(i,:) = stimulisig.C0(strcmp(thisstimuliword, stimulisig.name),:);
end

% do cut-off
wordsignalstrue = wordsignalstrue(:,1:cutoff)';

% create word-signals
[junk permorder] = sort(rand(20,nWords),2);
permorder = unique([1:nWords;permorder],'rows');
permorder(any(bsxfun(@minus,permorder, 1:nWords) == 0,2),:) = [];
permorder = [1:nWords;permorder];
[numberOfIterations ix] = size (permorder);
permorder = reshape(permorder',1, nWords*numberOfIterations);
wordsignals = zeros(cutoff,length(permorder));
for i = 1:length(permorder)
    wordsignals(:, i) = wordsignalstrue(:,permorder(i));
end
wordsignals = reshape(wordsignals, [cutoff nWords numberOfIterations]);
wordsignals = permute(wordsignals, [1 3 2]);
wordsignals = reshape(wordsignals, [cutoff*numberOfIterations nWords]);
wordsignals = repmat(wordsignals, 1, nSensors);

%restore shape
wordsignals = reshape(wordsignals, [cutoff numberOfIterations nWords*nSensors]);
wordsignals = permute(wordsignals, [1 3 2]);
wordsignals = reshape(wordsignals, [cutoff numberOfIterations*nWords*nSensors]);
wordsignals = single(wordsignals);
wordsignals(wordsignals==0) = nan;

% Convert to single object, in same order as wordlist

for i = 1:length(inputsarray) % for each word
    thisstimuliword = inputsarray{i,1};
    wordstructure{1,i} = thisstimuliword;
    eval([ 'wordstructure{2,i} = grandaverage_', thisstimuliword, '.time(:,:);' ]);
    eval([ 'wordstructure{3,i} = grandaverage_', thisstimuliword, '.avg(:,:);' ]);   
end

for i = 1:length(wordstructure(3,:))
    wordstructure{3,i} = wordstructure{3,i}';
end

clear('-regexp', '^grand');

allMEGdata = cell2mat(wordstructure(3,:));
allMEGdata = single(allMEGdata);

% generate RMS
for j = 3:3:length(allMEGdata)
    allMEGdata(:,j) = sqrt(allMEGdata(:,j-1).^2 + allMEGdata(:,j-2).^2);
end

nTotalColumns = nSensors*nWords;
initialPermutationSegment = 1:nSensors:nTotalColumns;
fullPermutation = repmat(initialPermutationSegment, 1, nSensors) + kron(0:nSensors-1, ones(1, nWords));
indicesMatrix = repmat(1:nTimePoints, 1, nTotalColumns) + kron((fullPermutation-1)*nTimePoints, ones(1, nTimePoints));
reorderedData = zeros(nTimePoints, nTotalColumns);
reorderedData(:) = allMEGdata(indicesMatrix(:));

allMEGdata = reorderedData;

clear reorderedData wordsignalstrue thisstimuliword stimulisig fid wordstructure inputsarray wordlistFilename wordlist fid times words indicesMatrix initialPermutationSegment fullPermutation inputfilename ix 


%parfor vectorised method===============================================
for q = 1:length(latencies);
    
    
    %-------------------------------
    % vectorised
    %-------------------------------
    
    % Get MEGdata for this latency
    latency = latencies(q);
    MEGdata = allMEGdata((200+latency+1):(200+cutoff+latency),:);
    MEGdata = repmat(MEGdata,1,numberOfIterations);
    
    % Average MEGdata for this latenecy
    h = ones(1,averagingwindow)/averagingwindow;
    %h = pdf('Normal',-floor(averagingwindow/2):floor(averagingwindow/2),0,1);   % gaussian
    MEGdata = filter(h, 1, MEGdata, [], 1);
        
    % Deal with if signals are shorter than the cut-off.
    MEGdata(isnan(wordsignals)) = NaN;

    % Do correlation
    [r c] = size(MEGdata);
    R = nansum((MEGdata-repmat(nanmean(MEGdata),r,1)).*(wordsignals-repmat(nanmean(wordsignals),r,1)))./((sqrt(nansum((MEGdata-repmat(nanmean(MEGdata), r, 1)).^2))).*(nansum((wordsignals-repmat(nanmean(wordsignals), r, 1)).^2)));
    
    % Split into relevent populations
    
    truewords = R(1:nSensors*nWords)';
    truewords = reshape(truewords,nWords,nSensors);
    truewords = truewords';
    randwords = R(((nSensors*nWords)+1):end)';
    randwords = reshape(randwords, nWords, nSensors, numberOfIterations-1);
    randwords = permute(randwords, [1 3 2]);
    randwords = reshape(randwords, nWords*(numberOfIterations-1), nSensors)';
    
    
    %-------------------------------
    % Transform populations with fisher-z
    %-------------------------------
    
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
    
    %to deal with positive and negitive
    for channel = 1:306
        if (mean(truewordsGuassian(channel,:)) < 0)
            truewordsGuassian(channel,:) = truewordsGuassian(channel,:).*-1;
        end
        if (mean(randwordsGuassian(channel,:)) < 0)
            randwordsGuassian(channel,:) = randwordsGuassian(channel,:)*-1;
        end
    end
    
    %lillefor test to check for Guassian
    
%     h = lillietest(randwordsGuassian(36,:));
%     
%     histfit(randwordsGuassian(42,:), 30); % plot
%     hold on;
%     histfit(truewordsGuassian(42,:), 30);
%     
%     h = findobj(gca,'Type', 'patch');
%     display(h);
%     set(h(1),'FaceColor',[0.982 0.909 0.721],'EdgeColor','k');
%     set(h(2),'FaceColor',[0.819 0.565 0.438],'EdgeColor','k');
    
    
    
    for k = 1:length(sigvalues)
        
        %-------------------------------
        % Do population test on signals
        %-------------------------------
        
        sigvalue = sigvalues(k);
        
        %make discete
        pvalues = [];
        for channel = 1:306
            truepopulation = truewordsGuassian(channel, :);
            randpopulation  = randwordsGuassian(channel, :);
            [h,p,ci,zval] = ztest(truepopulation,mean(randpopulation),std(randpopulation), sigvalue, 'right');
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
        
        col = pink(1024);    tmp = linspace(0,1,1024)';
        for n = 1:3, col(:,n) = interp1( 10.^tmp, col(:,n), 1+9*tmp, 'linear'); end
        
        cfg.comment         = [num2str(latency), 'ms'];
        cfg.commentpos      = 'middlebottom';
        cfg.colormap        = col;
        cfg.layout          = 'CBU_NM306mag.lay';
        cfg.colorbar        = 'WestOutside';        % outside left
        cfg.gridscale       = 140;                  % scaling grid size (default = 67)
        cfg.maplimits       = [0 1];                % Y-scale
        cfg.style           = 'both';               %(default)
        cfg.contournum      = 1;                    %(default = 6), see CONTOUR
        cfg.shading         = 'flat';               %(default = 'flat')
        cfg.contcolor       = [57 23 91];
        cfg.interpolation   = 'v4';                 % default, see GRIDDATA
        cfg.electrodes      = 'highlights';         % should be 'highlights' for white dots. But also 'off','labels','numbers','highlights' or 'dotnum' (default = 'on')
        cfg.ecolor          = [1 1 1];              % Marker color (default = [0 0 0] (black))
        cfg.highlight       = highlightchannels;    % or the channel numbers you want to highlight (default = 'off'). These numbers should correspond with the channels in the data, not in the layout file.
        cfg.hlcolor         = [1 1 1];
        
        topoplot(cfg, datavector)
        sigstring = num2str(sigvalue, '%6.6f');
        title([num2str(latency), 'ms'], 'Fontsize', 20, 'Fontweight', 'bold');
        outputfile = [outputpath, 'latency-', num2str(latency),'_pval-', sigstring, '.fig'];
        saveas(gcf, outputfile);
        close(gcf);
    end
    
end


