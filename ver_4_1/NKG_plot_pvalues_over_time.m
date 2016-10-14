
%import the results of a NKG anaylis with 2562 resolution, with lags
%between -200 to 800 and print out the p values over time.

% paths

addpath /imaging/at03/thirdparty_matlab_functions/;
addpath /imaging/at03/thirdparty_matlab_functions/export_fig/;

% variables

functionname = 'Glasberg-Moore Loudness';
alpha = 0.01;
ROInames = {
    'superiorfrontal'
    'superiorparietal'
    'superiortemporal'
    'middletemporal'
	'supramarginal'
    'temporalpole'
	'transversetemporal'
    'bankssts';
    'unknown'
    'caudalanteriorcingulate';
    'caudalmiddlefrontal';
    'corpuscallosum';
    'cuneus';
    'entorhinal';
    'frontalpole';
    'fusiform';
    'inferiorparietal'
    'inferiortemporal'
    'isthmuscingulate'
    'lateraloccipital'
    'lateralorbitofrontal'
    'medial_wall'
    'lingual'
    'medialorbitofrontal'
    'paracentral'
    'parahippocampal'
    'parsopercularis'
    'parsorbitalis'
    'parstriangularis'
    'pericalcarine'
	'postcentral'
    'posteriorcingulate'
    %'precentral'
    %'precuneus'
    %'rostralanteriorcingulate'
    %'rostralmiddlefrontal'
    };


%import data

import xxx;


% convert into pvalue-by-time

results = outputSTC.data';
results = results(:,1:5:end,:);
results(:,202:end) = [];

%define some variables

vertexorder = outputSTC.vertices;
timepoints = size(results,2);
numberofverts = size(results,1);

% Apply bonferroni correction (2 hemispheres x timepoints x no of sources)

results = results.*(2*timepoints*numberofverts);

%plot all

plot([-200:5:800], results');
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-200 800], 'YLim', [1E-10 2*timepoints*numberofverts]); %5E2
title(['vertex p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' resolution per hemisphere']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String',['Bonferroni-corrected p-value (with alpha at ', num2str(alpha) ')']);
h(1) = gridxy([], alpha ,'color','b','linestyle', ':');
h(2) = gridxy(0, [] ,'color','b','linestyle', ':');

export_fig(['p-values_for_' functionname '_' leftright, '_hemisphere_' experimentName '.png'], '-nocrop', '-a1') ;

%plot all movie frames

plot([-200:5:800], results');
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-200 800], 'YLim', [1E-10 2*timepoints*numberofverts]);
title(['vertex p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' resolution per hemisphere']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String',['p-value (with alpha at ', num2str(alpha) ')']);
h(1) = gridxy([], alpha ,'color','b','linestyle', ':');
h(2) = gridxy(0, [] ,'color','b','linestyle', ':');
h(3) = gridxy(1, [] ,'color','k','linestyle', '-');
for f = 1:timepoints
    delete(h(3));
    h(3) = gridxy((f*5)-205, [] ,'color','k','linestyle', '-');
    export_fig(['frame', num2str(f),'.jpg'], '-nocrop', '-a1') ;
end

%plot all as flowers

[minresults, positions] = min(results, [], 2);
flowerresults = ones(numberofverts, timepoints);
flowerresults(flowerresults == 1) = NaN;
for i = 1 :size(flowerresults, 1)
    flowerresults(i,positions(i)) = minresults(i);
end
stem([-200:5:800], flowerresults','fill', 'BaseValue', 2*timepoints*numberofverts);
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-200 800], 'YLim', [1E-10 2*timepoints*numberofverts]); %5E2
title(['vertex p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' resolution per hemisphere']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String',['Bonferroni-corrected p-value (with alpha at ', num2str(alpha) ')']);
h(1) = gridxy([], alpha ,'color','b','linestyle', ':');
h(2) = gridxy(0, [] ,'color','b','linestyle', ':');


%plot p-values by region of interest as mesh/image.

%loggedresults = 1-nthroot(results, 1);  % for image
loggedresults = results;
loggedresultsordered = zeros(numberofverts,timepoints);                    
count = 1;
%savenumberofvertsinROI = [];
for i = 1:length(ROInames)
    %numberofvertsinROI = 1;
    labelfilename = [rootDataSetPath, experimentName, '/nme_subject_dir/average/label/', ROInames{i}, '-', leftright, '.label'];
    fid = fopen(labelfilename);
    thisROI = textscan(fid, '%d %f %f %f %f', 'HeaderLines', 2);
    for v = 1:length(thisROI{1,1})    
        if thisROI{1,1}(v) <= (numberofverts-1)
            position = find(thisROI{1,1}(v) == vertexorder);
            loggedresultsordered(count,:) = loggedresults(position, :);
            count = count+1;
            %numberofvertsinROI = numberofvertsinROI+1;
        end
    end
    %savenumberofvertsinROI = [savenumberofvertsinROI numberofvertsinROI];
    fclose('all');
end
%imagesc([-200:5:800], 1:2562, loggedresultsordered);
%title(['vertex p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' spatial resolution ']);
%colormap('hot');
%set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');

smoothing=30;
loggedresultsordered(count+1:end,:) = [];
loggedresultsordered = conv2(loggedresultsordered, ones(smoothing,1)./9, 'same');
loggedresultsordered(end-smoothing:end,:) = [];
loggedresultsordered(1:smoothing,:) = [];
surf([-200:5:800], 1:size(loggedresultsordered,1), loggedresultsordered);
shading interp
set(gca,'ZDir','reverse', 'ZScale', 'log', 'XLim', [-200 800], 'ZLim', [1E-1 2]); % 2 because the smoothing moves it twoward 2 for some reason
title(['smoothed vertex p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' spatial resolution ']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'ZLabel'),'String','p-value');
set(get(gca,'YLabel'),'String','Vertex label grouped by ROI');
colormap('copper');
colormap(flipud(colormap))





%average p-values by region of interest.

ROIresults = zeros(length(ROInames),timepoints);


for i = 1:length(ROInames)
    count = 1;
    temp = [];
    labelfilename = [rootDataSetPath, experimentName, '/nme_subject_dir/average/label/', ROInames{i}, '-', leftright, '.label'];
    fid = fopen(labelfilename);
    thisROI = textscan(fid, '%d %f %f %f %f', 'HeaderLines', 2); 
    for v = 1:length(thisROI{1,1})
        if thisROI{1,1}(v) <= (numberofverts-1)
            position = find(thisROI{1,1}(v) == vertexorder);
            temp(count,:) = results(position, :);
            count = count+1;
        end
    end
    fclose('all');
    ROIresults(i, :) = mean(temp, 1);
end
   
%plot all

set(0,'DefaultAxesLineStyleOrder',{'-','--',':','-.'}, 'DefaultAxesColorOrder', distinguishable_colors(round(length(ROInames)/4)));
plot([-200:5:800], ROIresults', 'linewidth',2);
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-200 650]);
title(['ROI p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' spatial resolution ']);
columnlegend(3, ROInames, 'NorthEast');
set(get(gca,'XLabel'),'String','Lag reletive to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String','p-value');

%then export it
export_fig(['ROI_p-values_for_' functionname '_' leftright, '_hemisphere_' experimentName '.png'], '-nocrop', '-a1') ;









%do region of interest

leftright;

import ['filterRIO.' leftright];




%Subtract out vertices with wich we have low confidence:

outputSTCPvals.data(outputSTCClim.data < 0) = 1;
outputSTC = outputSTCPvals;

%seperate left & right sensors
load (datasets/verbphase/scripts/sensors_hemiphere_labels/)
resultsrh = results;
resultslh = results;
resultsrh(hems' ~= 0, :) = [];
resultslh(hems' ~= 1, :) = [];
