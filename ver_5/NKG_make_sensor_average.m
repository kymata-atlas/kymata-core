% NKG_make_sensorRMS average.m

temp = [];

parfor w = 1:numel(itemlist)
    
    thisword = char(itemlist(w));
    
    temp = [];
    
    sensordataveraged = fiff_read_evoked([rootCodeOutputPath version '/' experimentName, '/3-sensor-data/fif-out/meg15_allchan.fif']);
    
    
    
    for p = 1:numel(participentIDlist)
        
        if (exist([rootCodeOutputPath, version '/', experimentName, '/3-sensor-data/fif-out/', char(participentIDlist(p)), '_', thisword, '-ave.fif'], 'file'))
            
            sensordata = fiff_read_evoked([rootCodeOutputPath version '/' experimentName, '/3-sensor-data/fif-out/', char(participentIDlist(p)), '_', thisword, '-ave.fif']);
            
            % Average correct Mags and Grads
            
            for channel = 1:size(chnames,2)
                if ~isempty(find(strcmp(chnames(1,channel), sensordata.info.ch_names)));
                    temp(channel,1:size(sensordata.evoked.epochs,2), p) = sensordata.evoked.epochs(find(strcmp(chnames(1,channel), sensordata.info.ch_names)),1:size(sensordata.evoked.epochs,2));
                end
            end
            
            temp(temp == 0) = NaN;
            
        else
            % Nothing
        end
        
    end
    
    
    %average over participants
    temp = nanmean(temp, 3);
    
    
    % Create RMS in every 3rd slot (Over Mags)
    % for channel = 3:3:306
    %     temp(channel, :) = sqrt(temp(channel-2,:).^2 + temp(channel-1,:).^2);
    %end
    
    
    %delete grads
    %for channel = 305:-3:1
    %    temp(channel, :) = [];
    %end
    %for channel = 203:-2:1
    %    temp(channel, :) = [];
    %end
    
    % Save
    sensordataveraged.evoked.epochs(1:376,:) = temp;
    
    fiff_write_evoked([rootCodeOutputPath version '/' experimentName, '/3-sensor-data/fif-out-averaged-GRADS+MAGS-readyforNKG/', thisword, '.fif'], sensordataveraged);
    
    
    
    
end


