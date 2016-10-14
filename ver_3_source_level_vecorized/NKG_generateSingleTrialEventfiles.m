% create event and average files for single trials




for p = 1:numel(participentIDlist)

    for s = 1:participentSessionHash.get(num2str(participentIDlist(p)))

        %Load events for this session
        sessioneventsfilename = ['/imaging/at03/LexproMEG/meg08_0' num2str(participentIDlist(p)) '/meg08_0' num2str(participentIDlist(p)) '_part' num2str(s) '-acceptedwordevents.eve'];
        fid = fopen(sessioneventsfilename);
        events = textscan(fid, '%s %s %s %s %s', 'delimiter', '\t' );
        fclose('all');


        ht = java.util.Hashtable;
        
        for e = 2:length(events{1,5})
            
            thisword = events{1,5}{e};

            % Look in hash table to see if it is a repetition
            rep = 1;
            if (strcmp(ht.get(thisword), 'seen'))
                rep = 2;
            end 
            
            % write an eve file
            thisEveFileID = fopen(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/eventFiles/meg08_0' num2str(participentIDlist(p)) '_sess'  num2str(s) '-' thisword '-rep' num2str(rep) '.eve'], 'w');
            fprintf(thisEveFileID, [ ...
                '00000\t0.0\t0\t0\tpsuedo_event\n' ...
                events{1,1}{e} '\t' events{1,2}{e} '\t' events{1,3}{e} '\t' events{1,4}{e} '\t' events{1,5}{e} '\n' ...
            ]);
            
            
            % write an ave file
            thisAveFileID = fopen(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/aveFiles/meg08_0' num2str(participentIDlist(p)) '_sess'  num2str(s)  '-' thisword '-rep' num2str(rep) '.ave'], 'w');
            fprintf(thisAveFileID, [ ...
                'average {\n' ...
                'outfile /imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out/meg08_0' num2str(participentIDlist(p)) '_sess'  num2str(s) '-' thisword '-rep' num2str(rep) '.fif\n' ...
                'category {\n' ...
                'name "' thisword '"\n' ...
                'event ' num2str(events{1,4}{e}) '\n' ...
                'tmin -0.05\n' ...
                'tmax ' num2str((length_of_longest_stimuli + post_stimulus_window)/1000) '\n' ...
                'bmin -' num2str(pre_stimulus_window/1000) '\n' ...
                'bmax 0.0\n' ...
                'color 1 1 0\n' ...
                '}\n' ...
                '}\n' ...
                ]);               % 'gradReject 2000e-13\n magReject 4e-12\n eegReject 200e-6\n eogReject 200e-6\n
            fclose(thisAveFileID);
            
            %place in hash table so we know the next instance of this word
            %we see is a repetition
            ht.put(thisword, 'seen');
            

        end

    end

end
