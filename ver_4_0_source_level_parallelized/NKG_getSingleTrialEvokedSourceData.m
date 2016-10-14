%Split onto many machines - takes a long time!

for p = 1:numel(participentIDlist) 
    
    for w = 1:numel(wordlist) 
        
        thisword = char(wordlist(w));
        
        if  ~exist([rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/2-single-trial-source-data/averagemesh-vert2562-smooth5-elisinvsol+fixed+sLORETA/gave270psuedo/', char(participentIDlist(p)) '-' thisword '-rh.stc'], 'file')
            if exist([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/gave270psuedo/', char(participentIDlist(p)), '-', thisword, '.fif'], 'file')
                %system('setenv PATH /imaging/local/software/mne/mne_2.7.0/x86_64/mne/bin:$PATH')
                %system('hostname')
                
                unixCommand = ['mne_make_movie '];
                %unixCommand = [unixCommand '--inv /imaging/ef02/phrasal/' char(participentIDlist(p)) '/avg/' char(participentIDlist(p)) '_3L-loose0.2-depth-reg-inv.fif ' ];
                %unixCommand = [unixCommand '--inv /imaging/ef02/phrasal/' char(participentIDlist(p)) '/avg/' char(participentIDlist(p)) '_3L-loose0.2-depth-reg-inv.fif ' ];
                unixCommand = [unixCommand '--inv /imaging/at03/NKG_Data_Sets/VerbphraseMEG/' char(participentIDlist(p)) '/' char(participentIDlist(p)) '_3L-fixed-reg-inv.fif ' ];
                unixCommand = [unixCommand '--meas ' rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/gave270psuedo/', char(participentIDlist(p)), '-', thisword, '.fif ' ];
                unixCommand = [unixCommand '--morph average ' ];
                unixCommand = [unixCommand '--morphgrade 4 ' ]; % downsamples to 642 is 3
                parnum = char(participentIDlist(p));
                unixCommand = [unixCommand '--subject ' num2str(parnum(7:end)) ' ' ];
                unixCommand = [unixCommand '--stc ' rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/2-single-trial-source-data/averagemesh-vert2562-smooth5-elisinvsol+fixed+sLORETA/gave270psuedo/' char(participentIDlist(p)) '-' thisword ' '];
                unixCommand = [unixCommand '--smooth 5 ' ];
                unixCommand = [unixCommand '--bmin -' num2str(pre_stimulus_window) ' ' ];
                unixCommand = [unixCommand '--bmax 0 ' ];  
                unixCommand = [unixCommand '--picknormalcomp ' ];  
                unixCommand = [unixCommand '--sLORETA ' ];
                fprintf(['[unix:] ' unixCommand '\n']);
                unix(unixCommand);
                
            end
       end
    end
end



