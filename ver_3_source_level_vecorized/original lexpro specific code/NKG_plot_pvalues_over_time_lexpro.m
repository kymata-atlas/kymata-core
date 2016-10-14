
%import the results of a NKG anaylis with 2562 resolution, with lags
%between -200 to 800 and print out the p values over time.

% paths

addpath /imaging/at03/thirdparty_matlab_functions/;
addpath /imaging/at03/thirdparty_matlab_functions/export_fig/;

% variables

functionname = 'Cohort';
alpha = 0.001;
ROInames = {
    %'bankssts';
    %'caudalmiddlefrontal';
    %'fusiform';
    %'inferiorparietal'
    %'inferiortemporal'
    %'lateraloccipital'
    %'lateralorbitofrontal'
    %'medial_wall'
    %'middletemporal'
    %'parsopercularis'
    %'parsorbitalis'
    %'parstriangularis'
    %'postcentral'
    %'precentral'
    %'rostralmiddlefrontal'
    %'superiorfrontal'
    %'superiorparietal'
    %'superiortemporal'
	%'supramarginal'
	%'transversetemporal'
    %'unknown'
    %'a2005s-G_temp_sup-Planum_temp'
    'superiortemporal-ant'
    %'superiortemporal-post'
    %'middletemporal-ant'
    %'middletemporal-post'
    };


%import data

import xxx;


% convert into pvalue-by-time

results = outputrhcohort;
%results = results(:,1:5:end,:);
%results(:,67:end) = [];

%define some variables

vertexorder = outputSTCrh.vertices;
timepoints = size(results,2);
numberofverts = size(results,1);

%plot all

plot([-30:5:300], results');
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-30 300]);
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

set(0,'DefaultAxesLineStyleOrder',{'-','--',':','-.'}, 'DefaultAxesColorOrder', distinguishable_colors(round(length(ROInames)/3)));
plot([-200:5:800], ROIresults', 'linewidth',2);
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-200 650]);
title(['ROI p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' spatial resolution ']);
columnlegend(3, ROInames, 'NorthEast');
set(get(gca,'XLabel'),'String','Lag reletive to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String','p-value');

%then export it
export_fig(['ROI_p-values_for_' functionname '_' leftright, '_hemisphere_' experimentName '.png'], '-nocrop', '-a2') ;









%only return region of interest




ROIresults = zeros(length(ROInames),timepoints);

count = 1;
for i = 1:length(ROInames)
    labelfilename = [rootDataSetPath, experimentName, '/nme_subject_dir/average/label/', ROInames{i}, '-', leftright, '.label'];
    fid = fopen(labelfilename);
    thisROI = textscan(fid, '%d %f %f %f %f', 'HeaderLines', 2);
    for v = 1:length(thisROI{1,1})
        if thisROI{1,1}(v) <= (numberofverts-1)
            position = find(vertexorder == thisROI{1,1}(v));
            ROIresults(count,:) = results(position, :);
            count = count+1;
        end
    end
    fclose('all');
end

plot([-30:5:300],ROIresults)
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-30 300], 'YLim', [1E-5 1]);
title([' p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' spatial resolution ']);
set(get(gca,'XLabel'),'String','Lag reletive to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String','p-value');






%plot p-values by region of interest as mesh.

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


loggedresultsordered(count+1:end,:) = [];
mesh([-30:5:300], 1:count, loggedresultsordered);
set(gca,'ZDir','reverse', 'ZScale', 'log', 'XLim', [-30 300], 'YDir','reverse');
title(['vertex p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' spatial resolution ']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'ZLabel'),'String','p-value');
set(get(gca,'YLabel'),'String','Vertex label grouped by ROI');


