
% This gets rid of the sessions.

for p = 1:numel(participentIDlist)

    for w = 1:numel(wordlist)

        thisword = char(wordlist(w));
        seen = 'no';

        for s = 1:participentSessionHash.get(num2str(participentIDlist(p)))
            
            disp([num2str(s), '= s']);

            for rep = 1:2

                disp([num2str(rep), '= REP']);
                
                if exist(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out/meg08_0' num2str(participentIDlist(p)) '_sess' num2str(s) '-' thisword '-rep' num2str(rep) '.fif'], 'file')
                    if (strcmp(seen, 'yes'))
                        outputfile=['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-sorted/meg08_0' num2str(participentIDlist(p)) '-' thisword '-rep2.fif'];
                    else
                        seen = 'yes';
                        outputfile=['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-sorted/meg08_0' num2str(participentIDlist(p)) '-' thisword '-rep1.fif'];
                    end
                    copyfile(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out/meg08_0' num2str(participentIDlist(p)) '_sess' num2str(s) '-' thisword '-rep' num2str(rep) '.fif'],outputfile)
                end
 
            end

        end

    end

end

