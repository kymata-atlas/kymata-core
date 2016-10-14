%--------------------------
% Neurokymography on source
%--------------------------

% this script works on signals sampled at 1000 times a second, starting at
% t=1. AT


% Parallel===============================================================
if matlabpool('size') == 0
    matlabpool open 2
end


% Paths==================================================================
addpath /imaging/at03/NKG/
addpath /imaging/at03/thirdparty_matlab_functions/

% Global variables=======================================================
latencies = -200:5:800;                                                             % what latencies
cutoff = 6000;                                                                      % cutt-off for signal times
nWords = 135;                                                                       % number of words
nTimePoints = 7434;                                                                 % number of time points in raw data
nVertices = 128;                                                                  % number of sensors
numberOfIterations = 30; %30 = 
f_alpha = 0.001;
defaultdownsamplerate = 5;
functionname = 'loudness sones';

% Initialise=============================================================

% ====================Import raw data - averages=========================
% Import output template stc
% outputSTC = mne_read_stc_file([rootCodeOutputPath version '/' experimentName, '/4-averaged-by-trial-data/vert10242-smooth5-nodepth-corFM-snr1-signedwithminusequalszero/' char(wordlist(1)) '-' leftright '.stc']);
% outputSTC = rmfield(outputSTC, 'data');
outputSTC.data = zeros(nTimePoints, nVertices);

% Import source data
for w = 1:numel(estimulo) 
    
    thisword = char(estimulo(w));
    
    %inputfilename = [rootCodeOutputPath version '/' experimentName, '/4-averaged-by-trial-data/vert10242-smooth5-nodepth-corFM-snr1-signedwithminusequalszero/', thisword '-' leftright '.stc'];

    disp(num2str(w));
%    eval([ 'grandaverage_', thisword ' = interp1(grandaverage_', thisword, ', 5);' ]);
    for i = 1:128
        eval([ 'Grandaverage_', thisword, '(1:7260,i) = ctranspose(resample(double(grandaverage_', thisword, '(i,:)), 1000,512)));' ]);
    end
end

% % ====================Import raw data - individuals======================
% % Import output template stc
% outputSTC = mne_read_stc_file([rootCodeOutputPath version '/' experimentName, '/3-single-trial-source-data/vert2562-smooth5-nodepth-eliFM-snr1-signed/meg10_0007-', char(wordlist(1)) '-' leftright '.stc']);
% outputSTC = rmfield(outputSTC, 'data');
% outputSTC.data = zeros(nTimePoints, nVertices);
% 
% % Import source data
% for w = 1:numel(wordlist) 
%    
%     thisword = char(wordlist(w));
%     
%     inputfilename = [rootCodeOutputPath version '/' experimentName, '/3-single-trial-source-data/vert2562-smooth5-nodepth-eliFM-snr1-signed/meg10_0021-', thisword '-' leftright '.stc'];
% 
%     disp(num2str(w));
%     if (exist(inputfilename, 'file'))
%         sensordata = mne_read_stc_file(inputfilename);
%         eval([ 'grandaverage_', thisword ' = sensordata.data;' ]);
%     else
%         disp('no match')
%     end
% end

%=======================================================================

% import stimuli

%load([rootFunctionPath experimentName '/HTK-C0_deltaCO_deltadeltaC0/stimulisig.mat']);

% create word-signals true, in same order as wordlist

wordsignalstrue = zeros(numel(estimulo), size(stimulisig.loudnessinstantSones, 2));

for w = 1:numel(estimulo) 
        thisword = char(estimulo(w));
        wordsignalstrue(w,:) = stimulisig.loudnessinstantSones(strcmp(thisword, stimulisig.name),:);
end

% do cut-off
wordsignalstrue = wordsignalstrue(:,1:cutoff)';
wordsignalstrue = repmat(wordsignalstrue, [1 1 nVertices]);
wordsignalstrue = permute(wordsignalstrue, [3 1 2]);
wordsignalstrue = single(wordsignalstrue);
%wordsignalstrue(wordsignalstrue==0) = nan;

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

for w = 1:numel(estimulo) 
    thisword = char(estimulo(w));
    eval([ 'MEGsignalstrue(:,:,w) = ctranspose(Grandaverage_', thisword, '(:,:));' ]);  
end
allMEGdata = single(MEGsignalstrue);

% 
% for w = 1:numel(wordlist) 
%     thisword = char(wordlist(w));
%     
%     inputfilename = [rootCodeOutputPath version '/' experimentName, '/4-averaged-by-trial-data/vert10242-smooth5-nodepth-corFM-snr1-signedwithminusequalszero/', thisword '-' leftright '.stc'];
% 
%     disp(num2str(w));
%     if (exist(inputfilename, 'file'))
%         sensordata = mne_read_stc_file(inputfilename);
%         eval([ 'grandaverage_', thisword ' = sensordata.data;' ]);
%     end
%     
%     eval([ 'MEGsignalstrue(:,:,w) = single(grandaverage_', thisword, '(:,:));' ]);  
%     eval([ 'clear grandaverage_', thisword ]);  
% end
% allMEGdata = single(MEGsignalstrue);

clear('-regexp', '^grand');
clear MEGsignalstrue thisstimuliword stimulisig fid wordstructure inputsarray wordlistFilename wordlist fid times words indicesMatrix initialPermutationSegment fullPermutation inputfilename ix 
 

% Start Matching/Mismatching ===============================================


% do vectorisation for each timepoint
for q = 79:length(latencies);
    
    disp(['frame ', num2str(q) ' out of ' , num2str(length(latencies))]);
       
    % Get MEGdata for this latency
    latency = latencies(q);
    MEGdata = allMEGdata(:,(pre_stimulus_window+latency+1):(pre_stimulus_window+cutoff+latency),:);
        
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
    
    
%      %lillefor test to check for Guassian
%      
%           vertexnumber = 2444;
%      
%           h = lillietest(randwordsCorr(vertexnumber,:));
%      
%           histfit(randwordsCorr(vertexnumber,:), 40); % plot
%           hold on;
%           histfit(truewordsCorr(vertexnumber,:), 40);
%           h = findobj(gca,'Type', 'patch');
%           display(h);
%           set(h(1),'FaceColor',[0.982 0.909 0.721],'EdgeColor','k');
%           set(h(2),'FaceColor',[0.819 0.565 0.438],'EdgeColor','k');
    
    
    %-------------------------------
    % Do population test on signals
    %-------------------------------
    
    pvalues = zeros(1, nVertices);
    for vertex = 1:nVertices
        truepopulation = truewordsCorr(vertex, :);
        randpopulation  = randwordsCorr(vertex, :);

        
        
        % 2-sample t-test
        
        [h,p, ci, stats] = ttest2(truepopulation,randpopulation, 1-((1-f_alpha)^(1/(2*length(latencies)*nVertices))), 'right', 'unequal');
        pvalues(1, vertex) = p;
        
        
        % 2-sample rank-sum test
        
        %[p,h] = ranksum(truepopulation,randpopulation, 'alpha', 1-((1-f_alpha)^(1/(2*length(latencies)*nVertices))), 'tail', 'right');
        %pvalues(1, vertex) = p;
        
        % 2-sample t-test
        
        %[h,p, ci, stats] = ttest2(truepopulation,randpopulation, 1-((1-f_alpha)^(1/(2*length(latencies)*nVertices))), 'right', 'unequal');
        %pvalues(1, vertex) = ci(1);
    end
    
    %---------------------------------
    % Save at correct latency in STC
    %---------------------------------
    
    outputSTC.data((latency)+(pre_stimulus_window+1),:) = pvalues';
    
    clear MEGdata R randwordsGuassian truewordsGuassian truewords randwords;
    
end

clear MEGdataDownsampled  wordsignalstrueDownsampled allMEGdata 

matlabpool close;

time = toc;

 % rewrite for min-only movies
 results = outputSTC.data';
 results(results == 0) = 1;
 [minresults, positions] = min(results, [], 2);
 fullminresults = ones(numberofverts, size(outputSTC.data, 1));
 for i = 1 :size(fullminresults, 1)
     fullminresults(i,positions(i)) = minresults(i);
 end
 fullminresults(fullminresults >= 1-((1-alpha)^(1/(2*length(latencies)*nVertices)))) = 1;
 outputSTC.data = fullminresults';
 outputSTC.data(outputSTC.data ~= 1) = 0;
 outputSTC.data  = outputSTC.data' ;
 outputSTC.data = 1-  outputSTC.data;
 for j = 1 :1:size(outputSTC.data, 1)
    count = 10;
    for k = 1:5:size(outputSTC.data, 2)
        if outputSTC.data(j,k) == 1;
            outputSTC.data(j,k) = count; 
            count = count - 1;
        elseif (count >= 0 && count ~= 10)
            outputSTC.data(j,k) = count; 
            count = count - 1;
        end
    end
 end
outputSTC.data  = outputSTC.data' ;
 

% rewrite log scale for output
outputSTC2 = outputSTC;
%outputSTC2.data = nthroot(outputSTC2.data,4);
%outputSTC2.data = ones(nTimePoints, nVertices)-outputSTC2.data;
outputSTC2.data = outputSTC2.data' ;
outputSTC2.tmin = -0.2;
mne_write_stc_file( [rootCodeOutputPath version '/' experimentName, '/6-neurokymography-output/F0-upsampled-10bins-inversed-' leftright '.stc'], outputSTC2);


