
%import the results of a NKG anaylis with 2562 resolution, with lags
%between -200 to 800 and print out the p values over time.

% paths

addpath /imaging/at03/thirdparty_matlab_functions/;
addpath /imaging/at03/thirdparty_matlab_functions/export_fig/;

% variables

alpha = 1-normcdf(5,0,1); % 5-sigma


ROInames = {
%     'superiorfrontal'   % do APARC 2009
%     'superiorparietal'
%     'superiortemporal'
%     'middletemporal'
% 	  'supramarginal'
%     'temporalpole'
% 	  'transversetemporal'
%     'bankssts';
      'unknown'
 %    'caudalanteriorcingulate'
%     'caudalmiddlefrontal';
     'corpuscallosum'
%     'cuneus';
%     'entorhinal';
%     'frontalpole';
%     'fusiform';
%     'inferiorparietal'
%     'inferiortemporal'
     'isthmuscingulate'
%     'lateraloccipital'
%     'lateralorbitofrontal'
%     'medial_wall'
%     'lingual'
%     'medialorbitofrontal'
%     'paracentral'
     'parahippocampal'
%     'parsopercularis'
%     'parsorbitalis'
%     'parstriangularis'
%     'pericalcarine'
%     'postcentral'
%     'posteriorcingulate'
%     'precentral'
%     'precuneus'
%     'rostralanteriorcingulate'
%     'rostralmiddlefrontal'
    };

%import data

%import xxx;


% convert into pvalue-by-time

results = outputSTC.data';
results = results(:,1:5:end,:);
results(:,202:end) = [];

%define some variables

vertexorder = outputSTC.vertices;
timepoints = size(results,2);
numberofverts = size(results,1);

%remove unknown ROIs
% 
%labelfilename = [rootDataSetPath, '/DATASET_1-01_visual-only/mne_subjects_dir/0173/label/Destrieux_Atlas/Unknown-', leftright, '.label'];
%fid = fopen(labelfilename);
%thisROI = textscan(fid, '%d %f %f %f %f', 'HeaderLines', 2);
%for v = 1:length(thisROI{1,1})    
%      if thisROI{1,1}(v) <= (numberofverts-1)
%          position = find(thisROI{1,1}(v) == vertexorder);
%          results(position, :) = ones(1, size(results,2))-2;
%      end
%end
%fclose('all');

%export for Kymata new

% results = log10(results);
% results = single(results);
% results(isnan(results)) = 0; %///why is NaNs this??????
% fname = ['pvalues_' leftright '.ascii'];
% dlmwrite(fname, results,'delimiter', ',', 'precision', 4);

% Apply bonferroni correction (2 hemispheres x timepoints x no of sources)

%results = results.*(2*timepoints*numberofverts);
Bonalpha = 1-((1-alpha)^(1/(2*timepoints*numberofverts)));

%%%%%%%%%%%%%%%%%%%%
% Work out stem colours
%%%%%%%%%%%%%%%%%%%%

[minresults, positions] = min(results, [], 2);
flowerresults = ones(numberofverts, timepoints);
for i = 1:size(flowerresults, 1)
    flowerresults(i,positions(i)) = minresults(i);
    if ~strcmp(endtable.(leftright).name{i},functionname)
        flowerresults(i,:) = ones(1, timepoints);
    end
end
flowerresults(flowerresults == 1) = NaN;
stemcolors = zeros(size(flowerresults, 1),3);
for i = 1 :size(flowerresults, 1)
    if minresults(i) < Bonalpha
       stemcolors(i,:) = [1 0 0] ;  
    end
end
set(0,'DefaultAxesColorOrder', stemcolors);

%%%%%%%%%%%%%%%%%%%%
% Plot as Line Graphs
%%%%%%%%%%%%%%%%%%%%

%plot([-200:5:800], results', 'Color', [0.8 0.8 0.8]);
set(gca, 'YScale', 'log', 'XLim', [-200 600], 'YLim', [1E-60 1]);
if(strcmp(leftright,'lh'))
    set(gca,'YDir','reverse');
end
title(['vertex p-values for ' functionname ' over time (' leftright, ' hemisphere)']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String',['p-value (with alpha at ', num2str(alpha) ', Bonferroni corrected)']);
h(1) = gridxy([], Bonalpha ,'color','k','linestyle', ':');
h(2) = gridxy(0, [] ,'color','k','linestyle', ':');
hold on

%%%%%%%%%%%%%%%%%%%%
% Plot as flowers
%%%%%%%%%%%%%%%%%%%%

stem([-200:5:800], flowerresults','fill', 'BaseValue', 1);









%seperate left & right sensors
%load (datasets/verbphase/scripts/sensors_hemiphere_labels/)
%resultsrh = results;
%resultslh = results;
%resultsrh(hems' ~= 0, :) = [];
%resultslh(hems' ~= 1, :) = [];


results = outputSTC.data';
results = results(:,1:5:end,:);
results(:,202:end) = [];

%define some variables

vertexorder = outputSTC.vertices;
timepoints = size(results,2);
numberofverts = size(results,1);

results = log10(results);
results = single(results);
results(isnan(results)) = 0; %///why is NaNs this??????

fname = ['pvalues_' leftright '.dat'];
fid = fopen(fname, 'w');
% write 32-byte header (UTF-8)
header = 'kymata.pvalue';
encoded_str = unicode2native(header, 'UTF-8');
for i = 1:3
    encoded_str = [encoded_str 0]; 
end
encoded_str = [encoded_str 0 0 0 49];
for i = 1:12
    encoded_str = [encoded_str 0]; 
end
fwrite(fid, encoded_str, 'uint8');
% version header
fwrite(fid, results', 'float32');
fclose(fid);
