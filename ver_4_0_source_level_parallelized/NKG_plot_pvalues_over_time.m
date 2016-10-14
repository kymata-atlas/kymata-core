
%import the results of a NKG anaylis with 2562 resolution, with lags
%between -200 to 800 and print out the p values over time.

% paths

addpath /imaging/at03/thirdparty_matlab_functions/;
addpath /imaging/at03/thirdparty_matlab_functions/export_fig/;

% variables

functionname = 'CO';
alpha = 0.001;
ROInames = {
    'bankssts';
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
    'middletemporal'
    'paracentral'
    'parahippocampal'
    'parsopercularis'
    'parsorbitalis'
    'parstriangularis'
    'pericalcarine'
	'postcentral'
    'posteriorcingulate'
    'precentral'
    'precuneus'
    'rostralanteriorcingulate'
    'rostralmiddlefrontal'
    'superiorfrontal'
    'superiorparietal'
    'superiortemporal'
	'supramarginal'
    'temporalpole'
	'transversetemporal'
    'unknown'
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

%plot all

plot([-200:5:800], results');
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-200 800]);
title(['vertex p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' spatial resolution ']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String',['p-value (with alpha at ', num2str(alpha) ')']);
h(1) = gridxy([], alpha ,'color','b','linestyle', ':');

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
export_fig(['ROI_p-values_for_' functionname '_' leftright, '_hemisphere_' experimentName '.png'], '-nocrop', '-a2') ;









%do region of interest

leftright;

import ['filterRIO.' leftright];


