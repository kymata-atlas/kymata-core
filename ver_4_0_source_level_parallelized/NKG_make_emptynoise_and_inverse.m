unixCommand = ['setenv SUBJECTS_DIR /imaging/at03/LexproMEG/nme_subject_dir'];  % for some reasonnot working. Do from shell
fprintf(['[unix:] ' unixCommand '\n']);
unix(unixCommand);

% origional meg empty noise data can be found in
% /megdata/cbu/camtest/no_name/<date>

% elisabeths can be found in lexpro/er/.
 
% Get empty noise covarience martices and save them
     

for p = 1:numel(participentIDlist) 

        unixCommand = ['mne_process_raw '];
        unixCommand = [unixCommand '--raw /imaging/at03/LexproMEG/empty_room_recordings/0' num2str(participentIDlist(p)) '/emptyroom.fif ' ];
        unixCommand = [unixCommand '--cov /imaging/at03/LexproMEG/empty_room_recordings/emptyroom_cov_description_file.cov '];
        unixCommand = [unixCommand '--savecovtag ' num2str(participentIDlist(p)) '-cov '  ];
        unixCommand = [unixCommand '--projoff '  ];
        fprintf(['[unix:] ' unixCommand '\n']);
        unix(unixCommand);
end
    
    
% Create inverse solution

for p = 1:numel(participentIDlist)
            
        unixCommand = ['mne_do_inverse_operator '];
        unixCommand = [unixCommand '--fwd /imaging/ef02/lexpro/meg08_0' num2str(participentIDlist(p)) '/tr/meg08_0' num2str(participentIDlist(p)) '_5-3L-fwd.fif ' ];
        unixCommand = [unixCommand '--senscov /imaging/at03/LexproMEG/empty_room_recordings/0' num2str(participentIDlist(p)) '/emptyroom' num2str(participentIDlist(p)) '-cov.fif ' ];
        unixCommand = [unixCommand '--loose 0.2 '];
        unixCommand = [unixCommand '--depth '];
        %unixCommand = [unixCommand '--bad <name> ']; It takes the bad channels from inside the fif file.
        %unixCommand = [unixCommand '--fixed '];
        unixCommand = [unixCommand '--meg '];
        %unixCommand = [unixCommand '--eeg ']; % because their is no emptyroom eeg.
        unixCommand = [unixCommand '--fmrithresh 1 ']; %ignored
        unixCommand = [unixCommand '--inv /imaging/at03/LexproMEG/meg08_0' num2str(participentIDlist(p)) '/inv/meg08_0' num2str(participentIDlist(p)) '_3L-loose0.2-depth-reg-inv.fif' ];
        fprintf(['[unix:] ' unixCommand '\n']);
        unix(unixCommand);
end
