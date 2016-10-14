%--------------------------
% Neurokymography on source
%--------------------------

% this script works on signals sampled at 1000 times a second, starting at
% t=1. AT


% Parallel===============================================================
if matlabpool('size') == 0
    matlabpool open 8
end


% Paths==================================================================
addpath /imaging/at03/NKG/
addpath /imaging/at03/thirdparty_matlab_functions/

% Global variables=======================================================
latencies = -200:5:800;% -200:5:800;                                                % what latencies
cutoff = 1000;                                                                      % cutt-off for signal times
nWords = 480;                                                                       % number of words
nTimePoints = 2067;                                                                 % number of time points in raw data
nVertices = 2562; %642                                                              % number of sensors
leftright = 'rh';                                                                   % hemisphere
averagingwindow = 7; 
numberOfIterations = 30;
defaultdownsamplerate = 5;

% Initialise=============================================================

% ====================Import raw data - averages=========================
% Import output template stc
outputSTC = mne_read_stc_file([rootCodeOutputPath version '/' experimentName, '/3-averaged-by-trial-data/averagemesh-vert2562-smooth5-elisinvsol_snr1_nodepth_new/' char(wordlist(1)) '-' leftright '.stc']);
outputSTC = rmfield(outputSTC, 'data');
outputSTC.data = zeros(nTimePoints, nVertices);

% Import source data
for w = 1:numel(wordlist) 
    
    thisword = char(wordlist(w));
    
    inputfilename = [rootCodeOutputPath version '/' experimentName, '/3-averaged-by-trial-data/averagemesh-vert2562-smooth5-elisinvsol_snr1_nodepth_new/', thisword '-' leftright '.stc'];

    disp(num2str(w));
    if (exist(inputfilename, 'file'))
        sensordata = mne_read_stc_file(inputfilename);
        eval([ 'grandaverage_', thisword ' = sensordata.data;' ]);
    end
end

% % ====================Import raw data - individuals======================
% % Import output template stc
% outputSTC = mne_read_stc_file([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/2-single-trial-source-data/averagemesh-vert642-smooth5-elisinvsol/meg10_0007-', char(wordlist(1)) '-' leftright '.stc']);
% outputSTC = rmfield(outputSTC, 'data');
% outputSTC.data = zeros(nTimePoints, nVertices);
% 
% % Import source data
% for w = 1:numel(wordlist) 
%    
%     thisword = char(wordlist(w));
%     
%     inputfilename = [rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/2-single-trial-source-data/averagemesh-vert642-smooth5-elisinvsol/meg10_0007-', thisword '-' leftright '.stc'];
% 
%     disp(num2str(w));
%     if (exist(inputfilename, 'file'))
%         sensordata = mne_read_stc_file(inputfilename);
%         eval([ 'grandaverage_', thisword ' = sensordata.data;' ]);
%     end
% end

%=======================================================================

% import stimuli

%load([rootFunctionPath experimentName '/HTK-C0_deltaCO_deltadeltaC0/stimulisig.mat']);

% create word-signals true, in same order as wordlist

wordsignalstrue = zeros(numel(wordlist), size(stimulisig.pitchstrength, 2));

for w = 1:numel(wordlist) 
        thisword = char(wordlist(w));
        wordsignalstrue(w,:) = stimulisig.pitchstrength(strcmp(thisword, stimulisig.name),:);
end

% do cut-off
wordsignalstrue = wordsignalstrue(:,1:cutoff)';
wordsignalstrue = repmat(wordsignalstrue, [1 1 nVertices]);
wordsignalstrue = permute(wordsignalstrue, [3 1 2]);
wordsignalstrue = single(wordsignalstrue);
wordsignalstrue(wordsignalstrue==0) = nan;

% find lowest downsampling rate possible in the signals

%wordsignalstrueNy = wordsignalstrue+min(wordsignalstrue);
%wordsignalstrueNy(isnan(wordsignalstrueNy)) = [];
%dtmx=max(diff(wordsignalstrueNy));
%fNyq=1/(2*dtmx);
%downsamplerate = fNyq/100?;
%clear dtmx wordsignalstrueNy fNyq

downsamplerate = defaultdownsamplerate;

% create permorder
[junk permorder] = sort(rand(400,nWords),2);
permorder = unique([1:nWords;permorder],'rows');
permorder(any(bsxfun(@minus,permorder, 1:nWords) == 0,2),:) = [];
permorder = [1:nWords;permorder];
permorder = permorder(1:numberOfIterations+1,:);

% Convert MEG-signals true, in same order as wordlist

MEGsignalstrue = zeros(nVertices,nTimePoints,nWords);

for w = 1:numel(wordlist) 
    thisword = char(wordlist(w));
    eval([ 'MEGsignalstrue(:,:,w) = grandaverage_', thisword, '(:,:);' ]);  
    eval(['clear(''grandaverage_', thisword, ''');' ]);
end
allMEGdata = single(MEGsignalstrue);

clear MEGsignalstrue thisstimuliword stimulisig fid wordstructure inputsarray wordlistFilename wordlist fid times words indicesMatrix initialPermutationSegment fullPermutation inputfilename ix 
 

% Start Matching/Mismatching ===============================================


% do vectorisation for each timepoint
for q = 1:length(latencies);
    
    disp(['frame ', num2str(q) ' out of ' , num2str(length(latencies))]);
       
    % Get MEGdata for this latency
    latency = latencies(q);
    MEGdata = allMEGdata(:,(pre_stimulus_window+latency+1):(pre_stimulus_window+cutoff+latency),:);
    
        
    % Average MEGdata for this latenecy
    %h = ones(1,averagingwindow)/averagingwindow;
    %h = pdf('Normal',-floor(averagingwindow/2):floor(averagingwindow/2),0,1);   % gaussian
    %MEGdata = filter(h, 1, MEGdata, [], 1);
    
    % Deal with begining filtering problem.
    %MEGdata(1:averagingwindow,:) = [];
    %wordsignalstrue(1:averagingwindow,:) = [];
        
    %downsample
    MEGdataDownsampled = MEGdata(:,1:downsamplerate:end,:);
    wordsignalstrueDownsampled = wordsignalstrue(:,1:downsamplerate:end,:);
    
    %scatter(MEGdata(1,:,1),wordsignalstrue(1,:,1));
    %scatter(MEGdataDownsampled(1,:,1),wordsignalstrueDownsampled(1,:,1));
     
    %-------------------------------
    % Matched/Mismatched
    %-------------------------------
    
    % can this be GPUED????????????????????????????????????????????????????
    allcorrs = zeros(nVertices,numberOfIterations,nWords);
  
    parfor i = 1:numberOfIterations
        
        disp(['Iteration ', num2str(i) ' out of ' , num2str(numberOfIterations)]);
                
        shuffledMEGdata = permute(MEGdataDownsampled, [3 2 1]);
        downsampledtimespan = size(shuffledMEGdata, 2);
        shuffledMEGdata = reshape(shuffledMEGdata, nWords, nVertices*downsampledtimespan);
        shuffledMEGdata = shuffledMEGdata(permorder(i, :),:);
        shuffledMEGdata = reshape(shuffledMEGdata, nWords, downsampledtimespan, nVertices);
        shuffledMEGdata = permute(shuffledMEGdata, [3 2 1]);
        
        % Deal with if signals are shorter than the cut-off.
    	shuffledMEGdata(isnan(wordsignalstrueDownsampled)) = NaN;

        % Do correlation
        allcorrs(:,i,:) = nansum(bsxfun(@minus,shuffledMEGdata, nanmean(shuffledMEGdata,2)).*bsxfun(@minus,wordsignalstrueDownsampled, nanmean(wordsignalstrueDownsampled,2)),2)./(sqrt(nansum((bsxfun(@minus,shuffledMEGdata, nanmean(shuffledMEGdata, 2))).^2,2)).*sqrt(nansum((bsxfun(@minus,wordsignalstrueDownsampled, nanmean(wordsignalstrueDownsampled, 2))).^2,2)));
        
    end

    clear shuffledMEGdata;
    
    %-------------------------------
    % Transform populations with fisher-z
    %-------------------------------
    
    allcorrs = 0.5*log((1+allcorrs)./(1-allcorrs));
    
    truewordsCorr = reshape(allcorrs(:,1,:), nVertices, nWords, 1);
    randwordsCorr = reshape(allcorrs(:, 2:end, :), nVertices, (nWords*numberOfIterations)-nWords, 1);
         
    % % to deal with positive and negitive
    %for vertex = 1:nVertices
    %    if (mean(truewordsCorr(vertex,:)) < 0)
    %         truewordsCorr(vertex,:) = truewordsCorr(vertex,:).*-1;
    %    end
    %    if (mean(randwordsCorr(vertex,:)) < 0)
    %         randwordsCorr(vertex,:) = randwordsCorr(vertex,:)*-1;
    %    end
    %end
    
    
%     %lillefor test to check for Guassian
%     
%          h = lillietest(randwordsCorr(36,:));
%     
%          histfit(randwordsCorr(11,:), 30); % plot
%          hold on;
%          histfit(truewordsCorr(11,:), 30);
%          h = findobj(gca,'Type', 'patch');
%          display(h);
%          set(h(1),'FaceColor',[0.982 0.909 0.721],'EdgeColor','k');
%          set(h(2),'FaceColor',[0.819 0.565 0.438],'EdgeColor','k');
    
    
    %-------------------------------
    % Do population test on signals
    %-------------------------------
    
    pvalues = zeros(1, nVertices);
    for vertex = 1:nVertices
        truepopulation = truewordsCorr(vertex, :);
        randpopulation  = randwordsCorr(vertex, :);

        % 2-sample t-test
        
        [h,p, ci, stats] = ttest2(truepopulation,randpopulation, [], 'right', 'unequal');
        
        % Mann Whitney U - Results are very normal so don't need to use
        
        %[p, h, stats] = ranksum(truepopulation,randpopulation);
        %if(stats.zval < 0) % this means truepopulation < randpopulation
        %    p = 1-(p/2);
        %else
        %    p = p/2; % move from two sided to one sided test
        %end
        
        pvalues(1, vertex) = p;
    end
    
    %---------------------------------
    % Save at correct latency in STC
    %---------------------------------
    
    outputSTC.data((latency)+(pre_stimulus_window+1),:) = pvalues';
    
    clear MEGdata R randwordsGuassian truewordsGuassian truewords randwords;
    
end

matlabpool close;

time = toc;

% rewrite log scale for output
outputSTC.data = nthroot(outputSTC.data,3);

%numberofpositives = sum(outputSTC.data(:)==0);

outputSTC.data = ones(nTimePoints, nVertices)-outputSTC.data;
outputSTC.data = outputSTC.data' ;
outputSTC.tmin = -0.2;
mne_write_stc_file( [rootCodeOutputPath version '/' experimentName, '/5-neurokymography-output/averagemesh-vert2562-smooth5-elisinvsol/test-' leftright '.stc'], outputSTC);


