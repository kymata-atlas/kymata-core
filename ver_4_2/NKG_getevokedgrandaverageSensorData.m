
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% create grandaverage
%%%%%%%%%%%%%%%%%%%%%%%%%%

for p = 1:numel(participentIDlist)
    
    disp(['participant ' num2str(p)])
    
    for s = 1:participentSessionHash.get(char(participentIDlist(p)))
        
        disp(['...block ', num2str(s)])
        
        events = [];
        
        %Load events for this session
        sessioneventsfilename = [rootDataSetPath,  experimentName, '/', char(participentIDlist(p)), '/', char(participentIDlist(p)), '_part', num2str(s), '-acceptedwordevents.eve'];
        fid = fopen(sessioneventsfilename);
        events = textscan(fid, '%s %s %s %s %s', 'delimiter', '\t' );
        fclose('all');
        
        % write a grand average file
        thisAveFileID = fopen([rootCodeOutputPath, version, '/' experimentName, '/2-sensor-data/aveFiles/', char(participentIDlist(p)) '-grandave.ave'], 'w');
        avedescription = [ ...
            'average {\n' ...
            '\tgradReject 2000e-13\n\tmagReject 4e-12\n\teegReject 200e-6\n' ... %\teogReject 200e-6\n
            '\tlogfile ' rootCodeOutputPath, version, '/' experimentName, '/2-sensor-data/logs-fif-out-ave/' char(participentIDlist(p)) '-grandave.log\n' ... % NB: because there is only one .ave file for all parts, we can only specify one log file - but NME does put all the parts in this log file, it just overwrites them. You can use --savetagas in mne_process_raw to get around this.
            '\tcategory {\n' ...
            '\t\tname "average"\n'];
        
        for e = 2:length(events{1,5})
            
            avedescription = [avedescription, '\t\tevent ' events{1,4}{e} '\n'];
            
        end
        
        
        avedescription = [avedescription '\t\ttmin -' num2str(pre_stimulus_window/1000) '\n' ...
            '\t\ttmax ' num2str((length_of_longest_stimuli + post_stimulus_window)/1000) '\n' ...
            '\t\tbmin -' num2str(pre_stimulus_window/1000) '\n' ...
            '\t\tbmax 0.0\n' ...
            '\t\tcolor 1 1 0\n' ...
            '\t}\n}\n'];
        
        fprintf(thisAveFileID,avedescription);
        fclose(thisAveFileID);
        
       
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% make evoked grandaverage fif
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for p = 1:numel(participentIDlist) 
           

            unixCommand = ['mne_process_raw '];
            
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--raw ' rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/2-outputfiffiles/ICA_', char(participentIDlist(p)) '_part'  num2str(s)  '_sss_movecomp_tr.fif '];
                %unixCommand = [unixCommand '--raw /imaging/ef02/lexpro/' char(participentIDlist(p)) '/tr/' char(participentIDlist(p)) '_part'  num2str(s)  '_raw_sss_movecomp_tr.fif '];
            end
            
            unixCommand = [unixCommand '--lowpass ' num2str(temporal_downsampling_rate) ' ' ];
            
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--events ', rootCodeOutputPath, version '/' experimentName, '/2-sensor-data/eventFiles/', char(participentIDlist(p)) '_part' num2str(s) '.eve '];
            end
                
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--ave  ', rootCodeOutputPath, version '/' experimentName, '/2-sensor-data/aveFiles/', char(participentIDlist(p)) '-grandave.ave '];
            end
               
            unixCommand = [unixCommand '--gave  ', rootCodeOutputPath, version '/' experimentName, '/2-sensor-data/fif-out-averaged/', char(participentIDlist(p)) '-grandave.fif '];
            
            unixCommand = [unixCommand '--projoff'];
            fprintf(['[unix:] ' unixCommand '\n']);
            unix(unixCommand);
    
end


