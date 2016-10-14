
%import the results of a NKG anaylis with 2562 resolution, with lags
%between -200 to 800 and print out the p values over time.

% paths

addpath /imaging/at03/thirdparty_matlab_functions/;
addpath /imaging/at03/thirdparty_matlab_functions/export_fig/;

% variables

alpha = 1-normcdf(5,0,1); % 5-sigma
%alpha = 0.01;
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
results(isnan(results)) = 1;%replace Nans with 1s (for excluded labels)
%results(1:10122,:) = 1;
%results = results(:,40:70);

%export for Kymata

results = log10(results);
results = single(results);
%dlmwrite('pvalues_lh.ascii', results,'delimiter', '\t', 'precision', 4);
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

%define some variables

vertexorder = outputSTC.vertices;
timepoints = size(results,2);
numberofverts = size(results,1);

% Apply bonferroni correction (2 hemispheres x timepoints x no of sources)

%results = results.*(2*timepoints*numberofverts);
Bonalpha = 1-((1-alpha)^(1/(2*timepoints*numberofverts)));

% %remove unknown ROIs
% for i = 1:length(ROInames)
%     labelfilename = [rootDataSetPath, experimentName, '/nme_subject_dir/average/label/', ROInames{i}, '-', leftright, '.label'];
%     fid = fopen(labelfilename);
%     thisROI = textscan(fid, '%d %f %f %f %f', 'HeaderLines', 2);
%     for v = 1:length(thisROI{1,1})    
%         if thisROI{1,1}(v) <= (numberofverts-1)
%             position = find(thisROI{1,1}(v) == vertexorder);
%             results(position, :) = ones(1, size(results,2));
%         end
%     end
%     fclose('all');
% end

%%%%%%%%%%%%%%%%%%%%
% Work out stem colours
%%%%%%%%%%%%%%%%%%%%

[minresults, positions] = min(results, [], 2);
flowerresults = ones(numberofverts, timepoints);
for i = 1 :size(flowerresults, 1)
    flowerresults(i,positions(i)) = minresults(i);
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

plot([-200:5:800], results', 'Color', [0.8 0.8 0.8]);
set(gca, 'YScale', 'log', 'XLim', [-200 800], 'YLim', [1E-30 1]);
if(strcmp(leftright,'lh'))
    set(gca,'YDir','reverse');
end
%title(['vertex p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' resolution per hemisphere']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String',['p-value (with alpha at ', num2str(alpha) ', Bonferroni corrected)']);
h(1) = gridxy([], Bonalpha ,'color','b','linestyle', ':');
h(2) = gridxy(0, [] ,'color','b','linestyle', ':');
hold on

%%%%%%%%%%%%%%%%%%%%
% Plot as flowers
%%%%%%%%%%%%%%%%%%%%

stem([-200:5:800], flowerresults','fill', 'BaseValue', 1);









%%%%%%%%%%%%%%%%%%%%
% Others
%%%%%%%%%%%%%%%%%%%%


%do region of interest

leftright;

import ['filterRIO.' leftright];


%seperate left & right sensors
load (datasets/verbphase/scripts/sensors_hemiphere_labels/)
resultsrh = results;
resultslh = results;
resultsrh(hems' ~= 0, :) = [];
resultslh(hems' ~= 1, :) = [];
