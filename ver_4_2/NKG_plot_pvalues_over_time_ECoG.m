
%import the results of a NKG anaylis with 2562 resolution, with lags
%between -200 to 800 and print out the p values over time.

% added individual labeling of stems IZ 04/13

% paths

addpath /imaging/at03/thirdparty_matlab_functions/;
addpath /imaging/at03/thirdparty_matlab_functions/export_fig/;

% variables

alpha = 1-normcdf(5,0,1); % 5-sigma
alpha = 0.01;

leftright = 'lh'; 

%import data

%import xxx;

%in= load('data.mat');
%outputSTC = in.ACE_outputSTC;
%functionname = 'pitch';

% convert into pvalue-by-time

results = outputSTC.data';
results = results(:,1:5:end,:);
results(:,202:end) = [];


ch2 = ch;
if (strcmp(leftright,'rh'))
    results(1:62, :) = [];
    ch2(1:62) = [];
else
    results(63:end, :) = [];
    ch2(63:end) = [];
end

%define some variables

% vertexorder = outputSTC.vertices;
timepoints = size(results,2);
numberofverts = size(results,1);

% Apply bonferroni correction (2 hemispheres x timepoints x no of sources)

%results = results.*(2*timepoints*numberofverts);
Bonalpha = 1-((1-alpha)^(1/(timepoints*numberofverts)));

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
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-200 800], 'YLim', [1E-8 1]);
title(['vertex p-values for ' functionname ' over time for the ACE-ECoG data for x' num2str(numberofverts) ' electrodes ']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String',['p-value (with alpha at ', num2str(alpha) ', Bonferroni corrected)']);
h(1) = gridxy([], Bonalpha ,'color','b','linestyle', ':');
h(2) = gridxy(0, [] ,'color','b','linestyle', ':');
hold on

%%%%%%%%%%%%%%%%%%%%
% Plot as flowers
%%%%%%%%%%%%%%%%%%%%
s = stem([-200:5:800], flowerresults','fill', 'BaseValue', 1);
xd = get(s,'XData');
yd = get(s,'YData');
hold on
   
    current = flowerresults(i,:)';
    [r c] = find(flowerresults==current(~isnan(current)));
    
    for k=1:length(xd)
        
        current = flowerresults(k,:)';
        [r c] = find(flowerresults==current(~isnan(current)));
        
        %if k==31 || k==39 % broca's area
        %     text(xd{k}(c), yd{k}(c) *0.4, ch{r} , 'HorizontalAlignment' , 'center','Color','yellow');
        %else
            text(xd{k}(c), yd{k}(c) *0.4, ch2{r} , 'HorizontalAlignment' , 'center');
        %end

    end  
    

%%%%%%%%%%%%%%%%%%
% Plot all movie frames
%%%%%%%%%%%%%%%%%%

h(3) = gridxy(1, [] ,'color','k','linestyle', '-');
for f = 1:timepoints
    delete(h(3));
    h(3) = gridxy((f*5)-205, [] ,'color','k','linestyle', '-');
    export_fig(['frame', num2str(f),'.jpg'], '-nocrop', '-a1') ;
end



%plot all Confidence intervals

plot([-200:5:800], results');
set(gca, 'XLim', [-200 800], 'YLim', [-0.2 0.03]);
title(['vertex lower-bound confidence interval for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' resolution per hemisphere']);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String',['lower-bound confidence interval (with alpha at ', num2str(alpha) ', Bonferroni corrected)']);
h(1) = gridxy([], 0 ,'color','b','linestyle', ':');
h(2) = gridxy(0, [] ,'color','b','linestyle', ':');

export_fig(['lower-bound_confidence_interval_for_' functionname '_' leftright, '_hemisphere_' experimentName '.png'], '-nocrop', '-a1') ;


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
set(gca,'YDir','reverse', 'YScale', 'log', 'XLim', [-200 800]);
title(['ROI p-values for ' functionname ' over time (' leftright, ' hemisphere) for the ' experimentName ' data at x' num2str(numberofverts) ' spatial resolution ']);
columnlegend(3, ROInames, 'NorthEast');
set(get(gca,'XLabel'),'String','Lag reletive to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String','p-value');

%then export it
export_fig(['ROI_p-values_for_' functionname '_' leftright, '_hemisphere_' experimentName '.png'], '-nocrop', '-a1') ;









%do region of interest

leftright;

import ['filterRIO.' leftright];


%seperate left & right sensors
load (datasets/verbphase/scripts/sensors_hemiphere_labels/);
resultsrh = results;
resultslh = results;
resultsrh(hems' ~= 0, :) = [];
resultslh(hems' ~= 1, :) = [];
