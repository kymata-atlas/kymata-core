%Split onto many machines - takes a long time!

for p = 1:numel(participentIDlist) 
    
    for w = 1:numel(wordlist) 
        
        thisword = char(wordlist(w));
        
        if  ~exist([rootCodeOutputPath, version '/' experimentName, '/3-single-trial-source-data/vert2562-smooth5-nodepth-eliFM-snr1-EEGonly/', char(participentIDlist(p)) '-' thisword '-rh.stc'], 'file')
            if exist([rootCodeOutputPath version '/' experimentName, '/2-sensor-data/fif-out-averaged/', char(participentIDlist(p)), '-', thisword, '.fif'], 'file')
                
                unixCommand = ['mne_make_movie '];
                %unixCommand = [unixCommand '--inv /imaging/ef02/lexpro/' char(participentIDlist(p)) '/tr/' char(participentIDlist(p)) '_3L-loose0.2-reg-inv.fif ' ];
                unixCommand = [unixCommand '--inv ' rootCodeOutputPath version '/' experimentName, '/2-sensor-data/inverse-operators/' char(participentIDlist(p)) '_3L-loose0.2-nodepth-reg-EEGonly-inv.fif ' ];
                unixCommand = [unixCommand '--meas ' rootCodeOutputPath version '/' experimentName, '/2-sensor-data/fif-out-averaged/', char(participentIDlist(p)), '-', thisword, '.fif ' ];
                unixCommand = [unixCommand '--morph average ' ];
                unixCommand = [unixCommand '--morphgrade 4 ' ]; % downsamples to 642 is 3
                parnum = char(participentIDlist(p));
                unixCommand = [unixCommand '--subject ' num2str(parnum(7:end)) ' ' ];
                unixCommand = [unixCommand '--stc ' rootCodeOutputPath version '/' experimentName, '/3-single-trial-source-data/vert2562-smooth5-nodepth-eliFM-snr1-EEGonly/' char(participentIDlist(p)) '-' thisword ' '];
                unixCommand = [unixCommand '--smooth 5 ' ];
                unixCommand = [unixCommand '--snr 1 ' ];
                unixCommand = [unixCommand '--bmin -' num2str(pre_stimulus_window) ' ' ];
                unixCommand = [unixCommand '--bmax 0 ' ];  
                unixCommand = [unixCommand '--picknormalcomp ' ];  
                %unixCommand = [unixCommand '--sLORETA ' ];
                fprintf(['[unix:] ' unixCommand '\n']);
                unix(unixCommand);
                
            end
       end
    end
end



