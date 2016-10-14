%--------------------------
% Find corrolation
%--------------------------


addpath /opt/neuromag/meg_pd_1.2/
addpath z:/Fieldtrip/
addpath z:/Fieldtrip/template/layout/
addpath /opt/mne/matlab/toolbox/
addpath z:/Fieldtrip/fileio/

addpath /imaging/at03/Fieldtrip/
addpath /imaging/at03/Fieldtrip/template/
addpath /imaging/at03/Fieldtrip/fileio/
addpath /imaging/at03/Fieldtrip/
 
%---------------------------------
% Prepare for topoplot
%---------------------------------
for frame  = 901:5:1001
    MAGpvalues = [];
    highlightchannels = [];
    sigvalue = 0.000005;
    i=1;
    for channel = 3:3:306
        MAGpvalues =[MAGpvalues ; sensordata.evoked.epochs(channel, frame)];
        %if ((1-sensordata.evoked.epochs(channel, frame))<sigvalue)
        %   highlightchannels = [highlightchannels i];
        %end
        i=i+1;
    end
    
    datavector = MAGpvalues;
    
    %---------------------------------
    % Print in topoplot
    %---------------------------------
    
    cfg.colormap        = jet;
    %cfg.layout          = 'CBU_NM306mag.lay';
    cfg.layout          = 'NM306mag.lay';
    cfg.colorbar        = 'WestOutside';        % outside left
    cfg.gridscale       = 100;                  % scaling grid size (default = 67)
    cfg.maplimits       = [-6 6];  %[-6 6]      % Y-scale
    cfg.style           = 'both';               %(default)
    cfg.contournum      = 9;                    %(default = 6), see CONTOUR
    cfg.shading         = 'flat';               %(default = 'flat')
    cfg.interpolation   = 'v4';                 % default, see GRIDDATA
    cfg.electrodes      = 'highlights';         % should be 'highlights' for white dots. But also 'off','labels','numbers','highlights' or 'dotnum' (default = 'on')
    cfg.ecolor          = [0 0 0];              % Marker color (default = [0 0 0] (black))
    cfg.highlight       = highlightchannels;    % or the channel numbers you want to highlight (default = 'off'). These numbers should correspond with the channels in the data, not in the layout file.
    cfg.hlcolor         = [1 1 1];
    
    h=figure;

    topoplot(cfg, datavector)
   
    title([num2str(frame-pre_stimulus_window-1)  'ms'], 'fontsize', 15);
   
    
    filename = ['sensors-', num2str((frame+4)/5) ,'.jpg'];
   
    print(h,'-djpeg', filename);
   
    close(h);
    
end

