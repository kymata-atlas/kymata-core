
% initialise

windowbegining = 250;
windowend = 345;
ROInames = {
     'superiorfrontal'   % do APARC 2009
     'superiorparietal'
     %'superiortemporal'
     %'middletemporal'
     'supramarginal'
     'temporalpole'
 	 'transversetemporal'
     %'bankssts';
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
     'precentral'
     'precuneus'
     'rostralanteriorcingulate'
     'rostralmiddlefrontal'
    };
alpha = 1-normcdf(5,0,1); % 5-sigma

results_LH = outputSTC_LH.data';
vertexorder_LH = outputSTC_LH.vertices;
results_LH(results_LH ==0) = 1;
results_LH(results_LH > 1-((1-alpha)^(1/(2*201*2562)))) = 1;

results_RH = outputSTC_RH.data';
vertexorder_RH = outputSTC_RH.vertices;
results_RH(results_RH ==0) = 1;
results_RH(results_RH > 1-((1-alpha)^(1/(2*201*2562)))) = 1;


numberofverts = size(results_RH,1);

%remove unknown ROIs
for i = 1:length(ROInames)
    labelfilename = [rootDataSetPath, experimentName, '/nme_subject_dir/average/label/', ROInames{i}, '-rh.label'];
    fid = fopen(labelfilename);
    thisROI = textscan(fid, '%d %f %f %f %f', 'HeaderLines', 2);
    for v = 1:length(thisROI{1,1})
        if thisROI{1,1}(v) <= (numberofverts-1)
            position = find(thisROI{1,1}(v) == vertexorder_RH);
            results_RH(position, :) = ones(1, size(results_RH,2));
        end
    end
    fclose('all');
end

for i = 1:length(ROInames)
    labelfilename = [rootDataSetPath, experimentName, '/nme_subject_dir/average/label/', ROInames{i}, '-lh.label'];
    fid = fopen(labelfilename);
    thisROI = textscan(fid, '%d %f %f %f %f', 'HeaderLines', 2);
    for v = 1:length(thisROI{1,1})
        if thisROI{1,1}(v) <= (numberofverts-1)
            position = find(thisROI{1,1}(v) == vertexorder_LH);
            results_LH(position, :) = ones(1, size(results_LH,2));
        end
    end
    fclose('all');
end

%find vertex

[minresults_RH, positions_RH] = min(results_RH, [], 2);

RH_resultsvertices = [];
count = 1;
for i = 1:length(positions_RH)
    if positions_RH(i) ~= 1
        RH_resultsvertices(count,1) = positions_RH(i)-200;
        RH_resultsvertices(count,2) = vertexorder_RH(i);
    count = count + 1;
    end
end
       
[minresults_LH, positions_LH] = min(results_LH, [], 2);

LH_resultsvertices = [];
count = 1;
for i = 1:length(positions_LH)
    if positions_LH(i) ~= 1
        LH_resultsvertices(count,1) = positions_LH(i)-200;
        LH_resultsvertices(count,2) = vertexorder_LH(i);
    count = count + 1;
    end
end

Loudness_LH_resultsverticessubset = LH_resultsvertices;
Loudness_RH_resultsverticessubset = RH_resultsvertices;

for i = length(Loudness_RH_resultsverticessubset):-1:1
    if Loudness_RH_resultsverticessubset(i,1) < windowbegining || Loudness_RH_resultsverticessubset(i,1) > windowend
        Loudness_RH_resultsverticessubset(i,:) = [];
    end
end
for i = length(Loudness_LH_resultsverticessubset):-1:1
    if Loudness_LH_resultsverticessubset(i,1) < windowbegining || Loudness_LH_resultsverticessubset(i,1) > windowend
        Loudness_LH_resultsverticessubset(i,:) = [];
    end
end


%import surf file


[LH_verts,faces] = mne_read_surface('/imaging/at03/NKG_Data_Sets/VerbphraseMEG/nme_subject_dir/average/surf/lh.pial');
[RH_verts,faces] = mne_read_surface('/imaging/at03/NKG_Data_Sets/VerbphraseMEG/nme_subject_dir/average/surf/rh.pial');
for i = 1:length(Loudness_LH_resultsverticessubset)
    thisvertex = Loudness_LH_resultsverticessubset(i,2);
    Loudness_LH_resultsverticessubset(i,3) = LH_verts(thisvertex,2)*1000;
    Loudness_LH_resultsverticessubset(i,4) = LH_verts(thisvertex,3)*1000;
end
for i = 1:length(Loudness_RH_resultsverticessubset)
    thisvertex = Loudness_RH_resultsverticessubset(i,2);
    Loudness_RH_resultsverticessubset(i,3) = RH_verts(thisvertex,2)*1000;
    Loudness_RH_resultsverticessubset(i,4) = RH_verts(thisvertex,3)*1000;
end


% Find best fit + angle
[LHspatialfit LHresidual] = polyfit(Loudness_LH_resultsverticessubset(:,3),Loudness_LH_resultsverticessubset(:,4),1);
[RHspatialfit RHresidual] = polyfit(Loudness_RH_resultsverticessubset(:,3),Loudness_RH_resultsverticessubset(:,4),1);
%LHangle = arctan(abs(LHspatialfit(1))); % only in th symbolic maths toolbox?
LHangle = -24;
RHangle = -24;
Loudness_LH_resultsverticessubset(:,5) = (cosd(LHangle).*Loudness_LH_resultsverticessubset(:,3)) + (sind(LHangle).*Loudness_LH_resultsverticessubset(:,4));
Loudness_RH_resultsverticessubset(:,5) = (cosd(RHangle).*Loudness_RH_resultsverticessubset(:,3)) + (sind(RHangle).*Loudness_RH_resultsverticessubset(:,4));


% 
% % Seperate graphs
% 
% % Left-hand Graph
% scatter(Loudness_LH_resultsverticessubset(:,5),Loudness_LH_resultsverticessubset(:,1), 80 ,'o','r', 'filled');
% set(gca, 'XLim', [-40 0],'YLim', [240 360], 'YDir','reverse', 'XDir','reverse', 'XAxisLocation', 'top', 'YAxisLocation', 'right');
% hold;
% [LHfit LHresidual] = polyfit(Loudness_LH_resultsverticessubset(:,1),Loudness_LH_resultsverticessubset(:,5),1);
% plot([LHfit(2)+341*LHfit(1) LHfit(2)+276*LHfit(1)], [341 276]);
% grid on;
% set(get(gca,'YLabel'),'String','Lag relative to onset of stimuli (ms)');
% set(get(gca,'XLabel'),'String', ['MNI-space y co-ordinate location (mm)']);
% 
% % Right-hand Graph
% scatter(Loudness_RH_resultsverticessubset(:,5),Loudness_RH_resultsverticessubset(:,1), 80 ,'^','r', 'filled');
% hold;
% set(gca, 'XLim', [-40 0],'YLim', [240 360],  'XAxisLocation', 'top', 'YDir','reverse');
% [RHfit RHresidual] = polyfit(Loudness_RH_resultsverticessubset(:,1),Loudness_RH_resultsverticessubset(:,5),1);
% plot([RHfit(2)+261*RHfit(1) RHfit(2)+315*RHfit(1)], [261 315]);
% grid on;
% set(get(gca,'YLabel'),'String','Lag relative to onset of stimuli (ms)');
% set(get(gca,'XLabel'),'String', ['MNI-space y co-ordinate location (mm)']);

% Same graphs

% Left-hand Graph
scatter(Loudness_LH_resultsverticessubset(:,1), Loudness_LH_resultsverticessubset(:,5), 80 ,'o','r', 'filled');
set(gca, 'YLim', [-50 30],'XLim', [245 350]);
hold;
[LHfit LHresidual] = polyfit(Loudness_LH_resultsverticessubset(:,1),Loudness_LH_resultsverticessubset(:,5),1);
plot([261 341], [LHfit(2)+261*LHfit(1) LHfit(2)+341*LHfit(1)] );
grid on;
scatter(Loudness_RH_resultsverticessubset(:,1),Loudness_RH_resultsverticessubset(:,5), 80 ,'^','r', 'filled');
[RHfit RHresidual] = polyfit(Loudness_RH_resultsverticessubset(:,1),Loudness_RH_resultsverticessubset(:,5),1);
plot([256 316], [RHfit(2)+256*RHfit(1) RHfit(2)+315*RHfit(1)]);
set(get(gca,'XLabel'),'String','Lag relative to onset of stimuli (ms)');
set(get(gca,'YLabel'),'String', ['MNI-space y co-ordinate location (mm)']);




%plot([266 321],[LHfit(2)+266*LHfit(1) LHfit(2)+321*LHfit(1)]);
%plot([241 306],[RHfit(2)+241*RHfit(1) RHfit(2)+306*RHfit(1)]);

