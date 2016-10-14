
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% create event, average and cov files for single trials
%%%%%%%%%%%%%%%%%%%%%%%%%%

   
for p = 1:numel(participentIDlist)
    
    disp(['participant ' num2str(p)])
    
    for s = 1:participentNumBlockHash.get(char(participentIDlist(p)))
        
        disp(['...block ', num2str(s)])
        
        events = [];
        
        %Load events for this session
        sessioneventsfilename = [rootDataSetPath,  experimentName, '/', char(participentIDlist(p)), '/events_part', num2str(s), '_audio.eve'];
        fid = fopen(sessioneventsfilename);
        events = textscan(fid, '%s %s %s %s %s', 'delimiter', '\t' );
        fclose('all');
        
        % write an eve file
        thisEveFileID = fopen([rootCodeOutputPath, version, '/' experimentName, '/3-sensor-data/eventFiles/audio/', char(participentIDlist(p)), '_part', num2str(s), '.eve'], 'w');
        
        thisEveFile = [];
        
        %do first psuedo-event
        thisEveFile = [thisEveFile events{1,1}{1} '\t' events{1,3}{1} '\t' events{1,4}{1} '\n'];
        for e = 2:length(events{1,5})
            %Could use delay option, but I feel this way is safer as I can see what it is doing.
            thisEveFile = [thisEveFile num2str(str2num(events{1,1}{e})+audio_delivery_latency) '\t' events{1,3}{e} '\t' events{1,4}{e} '\n']    ;
        end
        fprintf(thisEveFileID, thisEveFile);
        fclose(thisEveFileID);
        

    end
    
end

