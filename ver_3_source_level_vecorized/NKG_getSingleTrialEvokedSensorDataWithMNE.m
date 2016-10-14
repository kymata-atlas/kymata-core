
for p = 1:numel(participentIDlist) % should be one
    
    for s = 1:participentSessionHash.get(num2str(participentIDlist(p)))
    
        for w = 1:numel(wordlist)
            
            thisword = char(wordlist(w));

            rep1filename = (['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/aveFiles/meg08_0' num2str(participentIDlist(p)) '_sess' num2str(s) '-' thisword '-rep1.ave']);
            rep2filename = ['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/aveFiles/meg08_0' num2str(participentIDlist(p)) '_sess' num2str(s) '-' thisword '-rep2.ave'];

            
            reparray = [];
            if exist(rep1filename, 'file')
                reparray = [1];
            elseif exist(rep2filename, 'file')
                reparray = [ 1 1 ];
            end
            if numel(reparray) > 0
                for r = 1:numel(reparray)
                    rep = r ;
                    unixCommand = ['mne_process_raw '];
                    unixCommand = [unixCommand '--raw /imaging/at03/LexproMEG/meg08_0' num2str(participentIDlist(p)) '/meg08_0' num2str(participentIDlist(p)) '_part'  num2str(s)  '_raw_sss_movecomp_tr.fif '];
                    unixCommand = [unixCommand '--lowpass ' num2str(temporal_downsampling_rate) ' ' ];
                    unixCommand = [unixCommand '--events /imaging/at03/NKG/saved_data/source_space/1-sensor-data/eventFiles/meg08_0' num2str(participentIDlist(p)) '_sess'  num2str(s)  '-' thisword '-rep' num2str(rep) '.eve '];
                    unixCommand = [unixCommand '--ave  /imaging/at03/NKG/saved_data/source_space/1-sensor-data/aveFiles/meg08_0' num2str(participentIDlist(p)) '_sess'  num2str(s)  '-' thisword '-rep' num2str(rep) '.ave '];
                    unixCommand = [unixCommand '--projoff ' ];
                    fprintf(['[unix:] ' unixCommand '\n']);
                    unix(unixCommand);

                end
                
            end
            
        end

    end

end