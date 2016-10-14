

%Run this a couple of times! Sometimes MNE fails.

for p = 1:numel(participentIDlist) % should be one
           
    for w = 1:numel(wordlist)
        
        thisword = char(wordlist(w));
        
        if ~exist([rootCodeOutputPath, version '/' experimentName, '/1-sensor-data/fif-out-averaged/', char(participentIDlist(p)) '-' thisword '.fif'],'file');
            
            unixCommand = ['mne_process_raw '];
            
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
                %unixCommand = [unixCommand '--raw /imaging/ef02/lexpro/' char(participentIDlist(p)) '/tr/', char(participentIDlist(p)) '_part'  num2str(s)  '_raw_sss_movecomp_tr.fif '];
                unixCommand = [unixCommand '--raw /imaging/ef02/phrasal/' char(participentIDlist(p)) '/tr/', char(participentIDlist(p)) '_part'  num2str(s)  '_raw_sss_movecomp_tr.fif '];
            end
            
            unixCommand = [unixCommand '--lowpass ' num2str(temporal_downsampling_rate) ' ' ];
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--events ', rootCodeOutputPath, version '/' experimentName, '/1-sensor-data/eventFiles/', char(participentIDlist(p)) '_part' num2str(s) '.eve '];
            end
                
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--ave  ', rootCodeOutputPath, version '/' experimentName, '/1-sensor-data/aveFiles/', char(participentIDlist(p)) '-' thisword '.ave '];
            end
                        
            unixCommand = [unixCommand '--gave  ', rootCodeOutputPath, version '/' experimentName, '/1-sensor-data/fif-out-averaged/', char(participentIDlist(p)) '-' thisword '.fif '];
            
            unixCommand = [unixCommand '--projoff'];
            fprintf(['[unix:] ' unixCommand '\n']);
            unix(unixCommand);
        end
        
    end
    
end








% seperate,
% for covarience matracies:


for p = 1:numel(participentIDlist) % should be one

            unixCommand = ['mne_process_raw '];
            
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
                %unixCommand = [unixCommand '--raw /imaging/ef02/lexpro/' char(participentIDlist(p)) '/tr/', char(participentIDlist(p)) '_part'  num2str(s)  '_raw_sss_movecomp_tr.fif '];
                unixCommand = [unixCommand '--raw /imaging/ef02/phrasal/' char(participentIDlist(p)) '/tr/', char(participentIDlist(p)) '_part'  num2str(s)  '_raw_sss_movecomp_tr.fif '];
            end
            
            unixCommand = [unixCommand '--lowpass ' num2str(temporal_downsampling_rate) ' ' ];
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--events ', rootCodeOutputPath, version '/' experimentName, '/1-sensor-data/eventFiles/', char(participentIDlist(p)) '_part' num2str(s) '.eve '];
            end

            
            for s = 1:participentSessionHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--cov  ', rootCodeOutputPath, version '/' experimentName, '/1-sensor-data/covFiles_usingbaseline/', char(participentIDlist(p)) '_part' num2str(s) '.cov '];
            end
            
            unixCommand = [unixCommand '--gcov  ', rootDataSetPath, experimentName, '/', char(participentIDlist(p)) ,'/' char(participentIDlist(p)), '_gcov.fif '];

            unixCommand = [unixCommand '--projoff'];
            fprintf(['[unix:] ' unixCommand '\n']);
            unix(unixCommand);
    
end