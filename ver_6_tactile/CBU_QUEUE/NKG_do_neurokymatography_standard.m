

%--------------------------
% Neurokymography on source
%--------------------------

% this script works on signals sampled at 1000 times a second, starting at
% t=1. AT

function NKG_do_neurokymatography_standard(leftright, functionname, functionlocation, stimulisigFunctionName, cutoff, nWords, nTimePoints, nVertices, outputfolder, inputfolder)

% setup file for neurokymatography scripts

% Add paths
addpath /imaging/local/linux/mne_2.6.0/mne/matlab/toolbox/;
addpath /imaging/at03/NKG_Code/Version6_tactile/mne_matlab_functions/;


%-------------------
% Set variables
%-------------------

% Root path variables
rootDataSetPath    = ['/imaging/at03/NKG_Data_Sets/'];
rootCodeOutputPath = ['/imaging/at03/NKG_Code_output/'];
rootFunctionPath   = ['/imaging/at03/NKG_Data_Functions/'];
version = 'Version5';


% Input variables

%Verbphrase
experimentName    = ['DATASET_3-02_tactile_toes']; 
itemlistFilename  = [rootDataSetPath, experimentName, '/items.txt'];
                                    
% MEG processing variables
pre_stimulus_window         = 200;                              % in milliseconds
post_stimulus_window        = 800;                              % in milliseconds
temporal_downsampling_rate  = 100;                              % in Hrz

% output variables
latency_step                = 5;                                % in milliseconds




%----------------------------------
% create wordlist
%----------------------------------

%create full wordlist (i.e. all words)

fid = fopen(itemlistFilename);
wordlist = textscan(fid, '%s');
wordlist = wordlist{1};
fclose('all');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Parallel===============================================================
% if matlabpool('size') == 0
%     matlabpool open 
% end



% Global variables=======================================================
latencies = -200:5:800;                                                             % what latencies                                                                % number of sensors
numberOfIterations = 5;       %30 = 
f_alpha = 0.001;
defaultdownsamplerate = 5;


% Initialise=============================================================


% Import output template stc
outputSTC = mne_read_stc_file([rootCodeOutputPath, version, '/', experimentName, '/5-averaged-by-trial-data/', inputfolder, char(wordlist(1)), '-', leftright, '.stc']);
outputSTC = rmfield(outputSTC, 'data');
outputSTC.data = zeros(nTimePoints, nVertices);


%=======================================================================

% import stimuli

load([rootFunctionPath 'DATASET_3-02_tactile' functionlocation 'stimulisig.mat']);
%load('/imaging/iz01/NeuroLex/NKG/erbscaledpitch/stimulisig.mat');
% create word-signals true, in same order as wordlist

eval(['wordsignalstrue = zeros(numel(wordlist), size(stimulisig.' stimulisigFunctionName ', 2));']);

for w = 1:numel(wordlist) 
        thisword = char(wordlist(w));
        eval(['wordsignalstrue(w,:) = stimulisig.' stimulisigFunctionName '(strcmp(thisword, stimulisig.name),:);']);
end

% do cut-off
wordsignalstrue = wordsignalstrue(:,1:cutoff)';
wordsignalstrue = repmat(wordsignalstrue, [1 1 nVertices]);
wordsignalstrue = permute(wordsignalstrue, [3 1 2]);
wordsignalstrue = single(wordsignalstrue);
wordsignalstrue = wordsignalstrue*-1;
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

allMEGdata = single(zeros(nVertices,nTimePoints,nWords));

for w = 1:numel(wordlist) 
    thisword = char(wordlist(w));
    
    inputfilename = [rootCodeOutputPath version '/' experimentName, '/5-averaged-by-trial-data/',inputfolder, '/', thisword '-' leftright '.stc'];

    disp(num2str(w));

    sensordata = mne_read_stc_file(inputfilename);
    allMEGdata(:,:,w) = sensordata.data(:,:);
end

clear('-regexp', '^grand');
clear MEGsignalstrue thisstimuliword stimulisig fid wordstructure inputsarray wordlistFilename wordlist fid times words indicesMatrix initialPermutationSegment fullPermutation inputfilename ix sensordata 
 

% Start Matching/Mismatching ===============================================


% do vectorisation for each timepoint
for q = 1:length(latencies);
    
    disp(['frame ', num2str(q) ' out of ' , num2str(length(latencies))]);
       
    % Get MEGdata for this latency
    latency = latencies(q);
    MEGdata = allMEGdata(:,(pre_stimulus_window+latency+1):(pre_stimulus_window+cutoff+latency),:);
        
    %downsample
    MEGdataDownsampled = MEGdata(:,1:downsamplerate:end,:);
    clear MEGdata;
    wordsignalstrueDownsampled = wordsignalstrue(:,1:downsamplerate:end,:);
    
    
    
    %scatter(MEGdata(1,:,1),wordsignalstrue(1,:,1));
    %scatter(MEGdataDownsampled(1,:,1),wordsignalstrueDownsampled(1,:,1));
     
    %-------------------------------
    % Matched/Mismatched
    %-------------------------------
    
    % can this be GPUED????????????????????????????????????????????????????
    allcorrs = zeros(nVertices,numberOfIterations,nWords);
  
    for i = 1:numberOfIterations
        
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
    
    %eliviates rounding of -0.999999 causing problems with log() in fisher-z transform.
    allcorrs(allcorrs<-0.999999) = -0.999999; 
    allcorrs(allcorrs>0.999999) = 0.999999;
    
    
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
    
    clear MEGdata wordsignalstrueDownsampled MEGdataDownsampled R randwordsGuassian truewordsGuassian truewords randwords;
    
end

%  matlabpool close;

functionname = [functionname '-flipped'];

cd(fullfile(rootCodeOutputPath, version, experimentName, '6-backedup-mat-files', outputfolder));
save('-v7.3',[functionname '_' leftright '_' num2str(nVertices) 'verts_-200-800ms_cuttoff' num2str(cutoff) '_' num2str(numberOfIterations) 'perms_ttestpval'], 'outputSTC', 'functionname', 'leftright', 'latencies', 'latency_step', 'nTimePoints', 'nVertices', 'nWords', 'numberOfIterations');

close all

  end