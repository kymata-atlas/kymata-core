
%create hash table containing number of sessions for each participant,
%in-case there are five sessions.

participentSessionHash = java.util.Hashtable;

for i = 1:numel(participentIDlist)
    eventfilename = [rootDataSetPath, experimentName, '/', char(participentIDlist(i)), '/', char(participentIDlist(i)), '_part' maxnumberofparts '-acceptedwordevents.eve'];
    if(exist(eventfilename, 'file'))
        participentSessionHash.put(char(participentIDlist(i)),maxnumberofparts);
    else
        participentSessionHash.put(char(participentIDlist(i)),maxnumberofparts-1);
    end
end



%if lexpro Use the 2.6.0 version of NME, and don't use psuedo_event header in event
%files

%if verbphrase Use the 2.7.0 version of NME, and use psuedo_event header in event
%files

% create event, average and cov files for single trials

for p = 1:numel(participentIDlist)
    
    
    for s = 1:participentSessionHash.get(char(participentIDlist(p)))
        
        events = [];
        
        %Load events for this session
        sessioneventsfilename = [rootDataSetPath,  experimentName, '/', char(participentIDlist(p)), '/', char(participentIDlist(p)), '_part', num2str(s), '-acceptedwordevents.eve'];
        fid = fopen(sessioneventsfilename);
        events = textscan(fid, '%s %s %s %s %s', 'delimiter', '\t' );
        fclose('all');
        
        %write an eve file
        thisEveFileID = fopen([rootCodeOutputPath, version, '/' experimentName, '/1-sensor-data/eventFiles/', char(participentIDlist(p)), '_part', num2str(s), '.eve'], 'w');
        
        thisEveFile = '00000\t0.0\t0\t0\tpsuedo_event\n'; % [];
        for e = 1:length(events{1,5})
            thisEveFile = [thisEveFile events{1,1}{e} '\t' events{1,2}{e} '\t' events{1,3}{e} '\t' events{1,4}{e} '\t' events{1,5}{e} '\n']    ;
        end
        fprintf(thisEveFileID, thisEveFile);
        fclose(thisEveFileID);
        
        
        % write a cov description file
        thisCovFileID = fopen([rootCodeOutputPath, version, '/' experimentName, '/1-sensor-data/covFiles_usingbaseline/', char(participentIDlist(p)) '_part', num2str(s), '.cov'], 'w');
        covdescription = [ ...
            'cov {\n' ...
            '\tgradReject 2000e-13\n\tmagReject 4e-12\n\teegReject 200e-6\n\teogReject 200e-6\n\toutfile /imaging/at03/' char(participentIDlist(p)) '_part', num2str(s), '_cov.fif\n\teventfile ', rootCodeOutputPath, version '/' experimentName, '/1-sensor-data/eventFiles/', char(participentIDlist(p)), '_part', num2str(s), '.eve\n\tkeepsamplemean\n' ...
            ];
        
        for e = 1:length(events{1,5})
            covdescription = [ covdescription, '\tdef {\n' ...
                '\t\tevent ' events{1,4}{e} '\n' ...
                '\t\ttmin -' num2str(pre_stimulus_window/1000) '\n' ...
                '\t\ttmax 0.0\n' ...
                '\t\tbmin -' num2str(pre_stimulus_window/1000) '\n' ...
                '\t\tbmax 0.0\n' ...
                '\t}\n' ...
                ];
        end
        
        covdescription = [covdescription '}\n'];
        fprintf(thisCovFileID,covdescription);
        fclose(thisCovFileID);
        
        
        
        for e = 1:length(events{1,5})
            
            thisword = char(events{1,5}{e});
            
            % write an ave file
            thisAveFileID = fopen([rootCodeOutputPath, version, '/' experimentName, '/1-sensor-data/aveFiles/', char(participentIDlist(p)) '-' thisword '.ave'], 'w');
            fprintf(thisAveFileID, [ ...
                'average {\n' ...
                '\tgradReject 2000e-13\n\tmagReject 4e-12\n\teegReject 200e-6\n\teogReject 200e-6\n' ...
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


