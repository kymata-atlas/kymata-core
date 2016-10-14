unixCommand = ['setenv SUBJECTS_DIR /imaging/at03/LexproMEG/nme_subject_dir'];  % for some reasonnot working. Do from shell
fprintf(['[unix:] ' unixCommand '\n']);
unix(unixCommand);

 
 
 % Spacially downsample STCs
     
 
for p = 1:numel(participentIDlist)
    
    for w = 1:numel(wordlist) 
        
        thisword = char(wordlist(w));
        
        unixCommand = ['mne_make_movie '];
        unixCommand = [unixCommand '--stcin /imaging/at03/NKG/saved_data/source_space/2-single-trial-source-data/meg08_0' num2str(participentIDlist(p)) '-' thisword ' ' ];
        unixCommand = [unixCommand '--morph average ' ]
        unixCommand = [unixCommand '--morphgrade 3 ' ]; % downsamples to 642 is 3, 2562 is 4
        unixCommand = [unixCommand '--subject 0' num2str(participentIDlist(p)) ' ' ];
        unixCommand = [unixCommand '--stc /imaging/at03/NKG/saved_data/source_space/3-averaged-by-trial-data/spatially_downsampled_ds-642/smooth3/elisabeth_invsol/0' num2str(participentIDlist(p)) '-' thisword ' '];
        unixCommand = [unixCommand '--smooth 3 ' ];
        fprintf(['[unix:] ' unixCommand '\n']);
        unix(unixCommand);
    end
end

%test = mne_read_stc_file(['/imaging/at03/NKG/saved_data/source_space/3-averaged-by-trial-data/spatially_downsampled_ds-2562/elisabeth_invsol/0320-bashed-rh.stc']);

% Average STCs
for w = 1:numel(wordlist)
    
    thisword = char(wordlist(w));
    disp([thisword]);
    temprh = [];
    templh = [];
    sensordatarh = [];
    sensordatalh = [];
    
    for p = 1:numel(participentIDlist)

        %lh-side
        lhfilename = ['/imaging/at03/NKG_Code_output/Version3_source_level_vecorized/3-averaged-by-trial-data/spatially_downsampled_ds-642/smooth5/elisabeth_invsol/', participentIDlist{p}(7:end) '-' thisword '-lh.stc'];
        if exist(lhfilename,'file')
            disp('lh');
            sensordatalh = mne_read_stc_file(lhfilename);
            
            templh(:,:,p) = sensordatalh.data(:, :);
        end
        
        %rh-side       
        rhfilename = ['/imaging/at03/NKG_Code_output/Version3_source_level_vecorized/3-averaged-by-trial-data/spatially_downsampled_ds-642/smooth5/elisabeth_invsol/' participentIDlist{p}(7:end) '-' thisword '-rh.stc'];
        if exist(rhfilename,'file')
            disp('rh');
            sensordatarh = mne_read_stc_file(rhfilename);
            
            temprh(:,:,p) = sensordatarh.data(:, :);
        end
    end
    
    %lh-side
    sensordatalh.data = mean(templh, 3);
    outputlh = ['/imaging/at03/NKG_Code_output/Version3_source_level_vecorized/3-averaged-by-trial-data/averaged_ds-642/smooth5/elisabeth_invsol/' thisword '-lh.stc'];
    mne_write_stc_file(outputlh, sensordatalh);
    
    %rh-side
    sensordatarh.data = mean(temprh, 3);
    outputrh = ['/imaging/at03/NKG_Code_output/Version3_source_level_vecorized/3-averaged-by-trial-data/averaged_ds-642/smooth5/elisabeth_invsol/' thisword '-rh.stc'];
    mne_write_stc_file(outputrh, sensordatarh);
    
end


