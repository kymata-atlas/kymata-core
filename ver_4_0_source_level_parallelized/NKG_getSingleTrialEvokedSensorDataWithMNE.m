

%Run this a couple of times! Sometimes MNE fails.

for p = 1:numel(participentIDlist) % should be one
    
    for s = 1:participentSessionHash.get(char(participentIDlist(p)))
    
        for w = 1:numel(wordlist)
            
            thisword = char(wordlist(w));

            rep1filename = ([rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/aveFiles/', char(participentIDlist(p)) '-' thisword '-rep1.ave']);
            rep2filename = ([rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/aveFiles/', char(participentIDlist(p)) '-' thisword '-rep2.ave']);

            
            reparray = [];
            if exist(rep1filename, 'file')
                reparray = [1];
            end
            if exist(rep2filename, 'file')
                reparray = [ 1 1 ];
            end
            if numel(reparray) > 0
                for r = 1:numel(reparray)
                    rep = r ;
                    if  ~exist([rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)) '-' thisword '-rep' num2str(rep) '.fif'], 'file')
                        unixCommand = ['mne_process_raw '];
                        %unixCommand = [unixCommand '--raw ' rootDataSetPath experimentName '/' char(participentIDlist(p)) '/', char(participentIDlist(p)) '_part'  num2str(s)  '_raw_sss_movecomp_tr.fif '];
                        unixCommand = [unixCommand '--raw /imaging/ef02/lexpro/' char(participentIDlist(p)) '/tr/', char(participentIDlist(p)) '_part'  num2str(s)  '_raw_sss_movecomp_tr.fif '];
                        unixCommand = [unixCommand '--lowpass ' num2str(temporal_downsampling_rate) ' ' ];
                        unixCommand = [unixCommand '--events ', rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/eventFiles/', char(participentIDlist(p)) '-' thisword '-rep' num2str(rep) '.eve '];
                        unixCommand = [unixCommand '--ave  ', rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/aveFiles/', char(participentIDlist(p)) '-' thisword '-rep' num2str(rep) '.ave '];
                        unixCommand = [unixCommand '--projoff >> ', rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)) '-' thisword '-rep' num2str(rep) '.log '];
                        fprintf(['[unix:] ' unixCommand '\n']);
                        unix(unixCommand);
                    end
                        
                end
                
            end
            
        end

    end

end