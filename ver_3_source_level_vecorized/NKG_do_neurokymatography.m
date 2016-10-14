%--------------------------
% Neurokymography on source
%--------------------------

% this script works on signals sampled at 1000 times a second, starting at
% t=1. AT


% Parallel===============================================================
%if matlabpool('size') == 0
%    matlabpool open 5
%end


% Paths==================================================================
addpath /imaging/at03/NKG_Code/Version3_source_level_vecorized/
addpath /imaging/at03/thirdparty_matlab_functions/

% Global variables=======================================================
latencies = 0:5:300;% -30:5:500;        50-110 and 600ms in a night                 % what latencies
cutoff = 700;                                                                       % cutt-off for signal times
nWords = 400;                                                                       % number of words
nTimePoints = 1301;                                                                 % number of time points in raw data
nVertices = 642; %2562                                                              % number of sensors
leftright = 'lh';                                                                   % hemisphere
downsamplerate = 5;
averagingwindow = 7;                                                                % this is not down sampled (was 7)

% Initialise=============================================================

% Import raw data - Output Neurokymography stc
outputSTC = mne_read_stc_file(['/imaging/at03/NKG_Code_output/Version3_source_level_vecorized/3-averaged-by-trial-data/averaged_ds-642/smooth5/elisabeth_invsol/bashed-' leftright '.stc']);
outputSTC = rmfield(outputSTC, 'data');
outputSTC.data = zeros(nTimePoints, nVertices);

% Import raw data - averages
for w = 1:numel(wordlist) 
    
    thisword = char(wordlist(w));
    
    inputfilename = ['/imaging/at03/NKG_Code_output/Version3_source_level_vecorized/3-averaged-by-trial-data/averaged_ds-642/smooth5/elisabeth_invsol/', thisword '-' leftright '.stc'];

    disp(num2str(w));
    if (exist(inputfilename, 'file'))
        sensordata = mne_read_stc_file(inputfilename);
        eval([ 'grandaverage_', thisword ' = sensordata.data;' ]);
    end
end


% import stimuli

load /imaging/at03/NKG/signals/C0_deltaC0_deltadeltaC0/stimulisig.mat;

% create word-signals true

wordsignalstrue = zeros(numel(wordlist), size(stimulisig.TriphoneAM250pruningNoLang, 2));

tic;

for w = 1:numel(wordlist) 
        thisword = char(wordlist(w));
        wordsignalstrue(w,:) = stimulisig.TriphoneAM250pruningNoLang(strcmp(thisword, stimulisig.name),:);
end

% do cut-off
wordsignalstrue = wordsignalstrue(:,1:cutoff)';

% create word-signals
[junk permorder] = sort(rand(20,nWords),2); %20 for eight
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
wordsignals = repmat(wordsignals, 1, nVertices);

%restore shape
wordsignals = reshape(wordsignals, [cutoff numberOfIterations nWords*nVertices]);
wordsignals = permute(wordsignals, [1 3 2]);
wordsignals = reshape(wordsignals, [cutoff numberOfIterations*nWords*nVertices]);
wordsignals = single(wordsignals);
wordsignals(wordsignals==0) = nan;

% Convert to single object, in same order as wordlist

for w = 1:numel(wordlist) 
    thisword = char(wordlist(w));
    wordstructure{1,w} = thisword;
    eval([ 'wordstructure{2,w} = grandaverage_', thisword, '(:,:);' ]);   
end

for i = 1:length(wordstructure(2,:))
    wordstructure{2,i} = wordstructure{2,i}';
end

clear('-regexp', '^grand');

allMEGdata = cell2mat(wordstructure(2,:));
allMEGdata = single(allMEGdata);

nTotalColumns = nVertices*nWords;
initialPermutationSegment = 1:nVertices:nTotalColumns;
fullPermutation = repmat(initialPermutationSegment, 1, nVertices) + kron(0:nVertices-1, ones(1, nWords));
indicesMatrix = repmat(1:nTimePoints, 1, nTotalColumns) + kron((fullPermutation-1)*nTimePoints, ones(1, nTimePoints));
reorderedData = zeros(nTimePoints, nTotalColumns);
reorderedData(:) = allMEGdata(indicesMatrix(:));

allMEGdata = reorderedData;

clear reorderedData wordsignalstrue thisstimuliword stimulisig fid wordstructure inputsarray wordlistFilename wordlist fid times words indicesMatrix initialPermutationSegment fullPermutation inputfilename ix 
 
 
%Downsample
%allMEGdata = downsample(allMEGdata, downsamplerate)';
%wordsignals = downsample(wordsignals, downsamplerate)';
%pre_stimulus_window = pre_stimulus_window/downsamplerate;
%latencies = latencies/downsamplerate;


% vectorised method===============================================


%THE BELOW IS NOW WRONG WHEN CHANGED FOR DOWNSAMPLINGG!!!!


% do vectorisation for each timepoint
for q = 1:length(latencies);
    
    disp(['frame ', num2str(q) ' out of ' , num2str(length(latencies))]);
    
    %-------------------------------
    % vectorised
    %-------------------------------
    
    % Get MEGdata for this latency
    latency = latencies(q);
    MEGdata = allMEGdata((pre_stimulus_window+latency+1):(pre_stimulus_window+cutoff+latency),:);
    MEGdata = repmat(MEGdata,1,numberOfIterations);
    
    % Average MEGdata for this latenecy
    h = ones(1,averagingwindow)/averagingwindow;
    %h = pdf('Normal',-floor(averagingwindow/2):floor(averagingwindow/2),0,1);   % gaussian
    MEGdata = filter(h, 1, MEGdata, [], 1);
        
    % Deal with if signals are shorter than the cut-off.
    MEGdata(isnan(wordsignals)) = NaN;
    
    % Deal with begining filtering problem.
    MEGdata(1:averagingwindow,:) = [];
    wordsignalstemp = wordsignals;
    wordsignalstemp(1:averagingwindow,:) = [];
    %scatter(MEGdata(:,2),wordsignalstemp(:,2));

    % Do correlation
    [r c] = size(MEGdata);
    R = nansum((MEGdata-repmat(nanmean(MEGdata),r,1)).*(wordsignalstemp-repmat(nanmean(wordsignalstemp),r,1)))./((sqrt(nansum((MEGdata-repmat(nanmean(MEGdata), r, 1)).^2))).*(nansum((wordsignalstemp-repmat(nanmean(wordsignalstemp), r, 1)).^2)));

    
    % Split into relevent populations
    
    truewords = R(1:nVertices*nWords)';
    truewords = reshape(truewords,nWords,nVertices);
    truewords = truewords';
    randwords = R(((nVertices*nWords)+1):end)';
    randwords = reshape(randwords, nWords, nVertices, numberOfIterations-1);
    randwords = permute(randwords, [1 3 2]);
    randwords = reshape(randwords, nWords*(numberOfIterations-1), nVertices)';
    
    
    %-------------------------------
    % Transform populations with fisher-z
    %-------------------------------
    
    randwordsGuassian = zeros(size(randwords, 1), size(randwords, 2));
    truewordsGuassian = zeros(size(truewords, 1), size(truewords, 2));
    
    for word = 1:size(randwords, 2)
        for channel = 1:nVertices
            r = randwords(channel,word);
            randwordsGuassian(channel,word) = 0.5*log((1+r)/(1-r));
        end
    end
    
    
    for word = 1:size(truewords, 2)
        for channel = 1:nVertices
            r = truewords(channel,word);
            truewordsGuassian(channel,word) = 0.5*log((1+r)/(1-r));
        end
    end
    
    %to deal with positive and negitive
    for channel = 1:nVertices
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
    %     histfit(randwordsGuassian(11,:), 30); % plot
    %     hold on;
    %     histfit(truewordsGuassian(11,:), 30);
    
    %     h = findobj(gca,'Type', 'patch');
    %     display(h);
    %     set(h(1),'FaceColor',[0.982 0.909 0.721],'EdgeColor','k');
    %     set(h(2),'FaceColor',[0.819 0.565 0.438],'EdgeColor','k');
    
    
    
    
    %-------------------------------
    % Do population test on signals
    %-------------------------------
    
    
    %make discete`
    pvalues = [];
    for channel = 1:nVertices
        truepopulation = truewordsGuassian(channel, :);
        randpopulation  = randwordsGuassian(channel, :);
        %[h,p,ci,zval] = ztest(truepopulation,mean(randpopulation),std(randpopulation), [], 'right');
        [h,p] = ttest2(truepopulation,randpopulation, [], 'right', 'unequal');
        pvalues = [pvalues p];
    end
    
    %---------------------------------
    % Save at correct latency in STC
    %---------------------------------
    
    outputSTC.data((latency)+(pre_stimulus_window+1),:) = pvalues';
    
   clear MEGdata R randwordsGuassian truewordsGuassian truewords randwords;
    
end

time = toc;

% rewrite for output
for k = 1:size(outputSTC.data, 1)
   for l = 1:size(outputSTC.data, 2)
        if (outputSTC.data(k,l) > 0.05)
           outputSTC.data(k,l) = 1;
        end
        if (outputSTC.data(k,l) <= 0.05 && outputSTC.data(k,l) > 0.01)
            outputSTC.data(k,l) = 0.1;
        end
        if (outputSTC.data(k,l) <= 0.01 && outputSTC.data(k,l) > 0)
            outputSTC.data(k,l) = 0;
        end
    end
end

%for k = 1:size(outputSTC.data, 1)
%   for l = 1:size(outputSTC.data, 2)
%        if outputSTC.data(k,l) < 0.05 && outputSTC.data(k,l) ~= 0  
%           outputSTC.data(k,l) = 0;
%        else
%           outputSTC.data(k,l) = NaN;
%        end
%    end
%end

%numberofpositives = sum(outputSTC.data(:)==0);

outputSTC.data = ones(1301, nVertices)-outputSTC.data;
outputSTC.data = outputSTC.data' ;
outputSTC.tmin = -0.1;
mne_write_stc_file( ['/imaging/at03/NKG_Code_output/Version3_source_level_vecorized/4-neurokymatogaphy-output/elisabeth_invsol/test-' leftright '.stc'], outputSTC);


