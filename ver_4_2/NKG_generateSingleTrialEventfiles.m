

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% make event files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for p = 1:numel(participentIDlist)

            unixCommand = ['mne_process_raw '];
            
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
               unixCommand = [unixCommand '--raw ' rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/2-outputfiffiles/ICA_', char(participentIDlist(p)) '_part'  num2str(s)  '_sss_movecomp_tr.fif '];
            end
            
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))   
                unixCommand = [unixCommand '--eventsout ', rootCodeOutputPath, version '/' experimentName, '/2-sensor-data/eventFiles/', char(participentIDlist(p)) '_part'  num2str(s) '-2_7_3.eve '];
            end
            
            fprintf(['[unix:] ' unixCommand '\n']);
            unix(unixCommand);
    
end


% Do preprocessing on event files

% in seperate perl scripts

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% create event, average and cov files for single trials
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
        
        % write an eve file
         thisEveFileID = fopen([rootCodeOutputPath, version, '/' experimentName, '/2-sensor-data/eventFiles/', char(participentIDlist(p)), '_part', num2str(s), '.eve'], 'w');
         
         thisEveFile = [];
        
         %do first psuedo-event
         thisEveFile = [thisEveFile events{1,1}{1} '\t' events{1,2}{1} '\t' events{1,3}{1} '\t' events{1,4}{1} '\t' events{1,5}{1} '\n'];
         for e = 2:length(events{1,5})
             %Could use delay option, but I feel this way is safer as I can see what it is doing.
             thisEveFile = [thisEveFile num2str(str2num(events{1,1}{e})+audio_delivery_latency) '\t' num2str(str2num(events{1,2}{e})+(audio_delivery_latency/1000)) '\t' events{1,3}{e} '\t' events{1,4}{e} '\t' events{1,5}{e} '\n']    ;
         end
         fprintf(thisEveFileID, thisEveFile);
         fclose(thisEveFileID);
        
        
        % write a cov description file
        thisCovFileID = fopen([rootCodeOutputPath, version, '/' experimentName, '/2-sensor-data/cov-description-files/', char(participentIDlist(p)) '_part', num2str(s), '.cov'], 'w');
        covdescription = [ ...
            'cov {\n' ...
            '\tgradReject 2000e-13\n\tmagReject 4e-12\n\teegReject 200e-6\n' ...
            '\tkeepsamplemean\n' ...
            '\teventfile '  rootCodeOutputPath, version, '/' experimentName, '/2-sensor-data/eventFiles/' char(participentIDlist(p)), '_part', num2str(s), '.eve\n' ...
            '\toutfile ' rootCodeOutputPath, version, '/' experimentName, '/2-sensor-data/noise-covarience-files-before-averaging/' char(participentIDlist(p)) '_part', num2str(s), '_cov.fif\n' ...
            '\tlogfile ' rootCodeOutputPath, version, '/' experimentName, '/2-sensor-data/logs-noise-covarience/' char(participentIDlist(p)), '_part', num2str(s), '_cov.log\n' ...
            ];
        
        for e = 2:length(events{1,5})
            covdescription = [ covdescription, '\tdef {\n' ...
                '\t\tevent ' events{1,4}{e} '\n' ...
                '\t\ttmin -' num2str(pre_stimulus_window/1000) '\n' ...
                '\t\ttmax 0.0\n' ...
                '\t\tbmin -' num2str(pre_stimulus_window/1000 + 0.2) '\n' ...
                '\t\tbmax -' num2str(pre_stimulus_window/1000) '\n' ...
                '\t}\n' ...
                ];
        end
        
        covdescription = [covdescription '}\n'];
        fprintf(thisCovFileID,covdescription);
        fclose(thisCovFileID);
        
        
        
        for e = 2:length(events{1,5})
            
            thisword = char(events{1,5}{e});
            
            % write an ave file
            thisAveFileID = fopen([rootCodeOutputPath, version, '/' experimentName, '/2-sensor-data/aveFiles/', char(participentIDlist(p)) '-' thisword '.ave'], 'w');
            fprintf(thisAveFileID, [ ...
                'average {\n' ...
                '\tgradReject 2000e-13\n\tmagReject 4e-12\n\teegReject 200e-6\n' ... %\teogReject 200e-6\n    
                '\tlogfile ' rootCodeOutputPath, version, '/' experimentName, '/2-sensor-data/logs-fif-out-ave/' char(participentIDlist(p)) '-' thisword '_ave.log\n' ... % NB: because there is only one .ave file for all parts, we can only specify one log file - but NME does put all the parts in this log file, it just overwrites them. You can use --savetagas in mne_process_raw to get around this.
                '\tcategory {\n' ...
                '\t\tname "' thisword '"\n' ...
                '\t\tevent ' events{1,4}{e} '\n' ...
                '\t\ttmin -' num2str(pre_stimulus_window/1000) '\n' ...
                '\t\ttmax ' num2str((length_of_longest_stimuli + post_stimulus_window)/1000) '\n' ...
                '\t\tbmin -' num2str(pre_stimulus_window/1000) '\n' ...
                '\t\tbmax 0.0\n' ...
                '\t\tcolor 1 1 0\n' ...
                '\t}\n' ...
                '}\n' ...
                ]);
            fclose(thisAveFileID);
            
            
            
        end
        
    end
end

