
%create hash table containing number of sessions for each participant,
%in-case there are five sessions.

participentSessionHash = java.util.Hashtable;

for i = 1:numel(participentIDlist)
    eventfilename = [rootDataSetPath, experimentName, '/', char(participentIDlist(i)), '/', char(participentIDlist(i)), '_part5-acceptedwordevents.eve'];
    if(exist(eventfilename, 'file'))
        participentSessionHash.put(char(participentIDlist(i)),5);
    else
        participentSessionHash.put(char(participentIDlist(i)),4);
    end
end



% create event and average files for single trials

for p = 1:numel(participentIDlist)

    ht = java.util.Hashtable;

    for s = 1:participentSessionHash.get(char(participentIDlist(p)))

        %Load events for this session
        sessioneventsfilename = [rootDataSetPath,  experimentName, '/', char(participentIDlist(p)), '/', char(participentIDlist(p)), '_part', num2str(s), '-acceptedwordevents.eve'];
        fid = fopen(sessioneventsfilename);
        events = textscan(fid, '%s %s %s %s %s', 'delimiter', '\t' );
        fclose('all');


        
        
        for e = 2:length(events{1,5})
            
            thisword = events{1,5}{e};

            % Look in hash table to see if it is a repetition
            rep = 1;
            if (strcmp(ht.get(thisword), 'seen'))
                rep = 2;

                disp(['SEEN!', num2str(rep)]); % for some completely unknown reason if this isn't there then it doesn't notice the contents of this loop!
            end 
            
            % write an eve file
            thisEveFileID = fopen([rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/eventFiles/', char(participentIDlist(p)) '-' thisword '-rep' num2str(rep) '.eve'], 'w');
            fprintf(thisEveFileID, [ ...
                '00000\t0.0\t0\t0\tpsuedo_event\n' ...
                events{1,1}{e} '\t' events{1,2}{e} '\t' events{1,3}{e} '\t' events{1,4}{e} '\t' events{1,5}{e} '\n' ...
            ]);
            
            
            % write an ave file
            thisAveFileID = fopen([rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/aveFiles/', char(participentIDlist(p)) '-' thisword '-rep' num2str(rep) '.ave'], 'w');
            fprintf(thisAveFileID, [ ...
                'average {\n' ...
                'gradReject 2000e-13\n magReject 4e-12\n eegReject 200e-6\n eogReject 200e-6\n' ...
                'outfile ' rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)) '-' thisword '-rep' num2str(rep) '.fif\n' ...
                'logfile ' rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)) '-' thisword '-rep' num2str(rep) '.log\n' ...
                'category {\n' ...
                'name "' thisword '"\n' ...
                'event ' num2str(events{1,4}{e}) '\n' ...
                'tmin -' num2str(pre_stimulus_window/1000) '\n' ...
                'tmax ' num2str((length_of_longest_stimuli + post_stimulus_window)/1000) '\n' ...
                'bmin -' num2str(pre_stimulus_window/1000) '\n' ...
                'bmax 0.0\n' ...
                'color 1 1 0\n' ...
                '}\n' ...
                '}\n' ...
                ]);              
            fclose(thisAveFileID);
            
            %place in hash table so we know the next instance of this word
            %we see is a repetition
            ht.put(thisword, 'seen');
            

        end

    end

end
