
    %make morph-maps for all subjects in the subjects durectory
    %to the average (no 'from' is specified) and save them as
    %<nme_subject_directory>/morph-maps/'subject'-average-morph.fif
    
    cd /imaging/at03/NKG_Data_Sets/VerbphraseMEG/nme_subject_dir
    
    %make sure 'setenv SUBJECTS_DIR
    %/imaging/at03/xxxxxxx/nme_subject_dir' is set correctly!
        
    unixCommand = ['mne_make_morph_maps '];
    unixCommand = [unixCommand '--to average '];
    unixCommand = [unixCommand '--redo ' ];
    unixCommand = [unixCommand '--all ' ];
    fprintf(['[unix:] ' unixCommand '\n']);
    unix(unixCommand);