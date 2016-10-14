
    %make morph-maps for all subjects in the subjects durectory
    %to the average (no 'from' is specified) and save them as
    %<nme_subject_directory>/morph-maps/'subject'-average-morph.fif
    
    %make sure 'setenv SUBJECTS_DIR
    %/imaging/at03/xxxxxxx/nme_subject_dir' is set correctly!
    
    % the following command takes about 30 mins
    make_average_subject --subjects 0173 0178 0436 0191 0193 0213 0219 0226 0230 0231 0239 0195
    
   
        
    mne_make_morph_maps --to average --redo --all