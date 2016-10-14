
% Average STCs
for w = 1:numel(wordlist)
    
    thisword = char(wordlist(w));
    
    if ~exist([rootCodeOutputPath version '/' experimentName, '/4-averaged-by-trial-data/vert2562-smooth5-depth-corFM-snr1-signed/' thisword '-rh.stc'],'file')
        
        disp([thisword]);
        temprh = [];
        templh = [];
        sensordatarh = [];
        sensordatalh = [];
        
        for p = 1:numel(participentIDlist)
            
            disp(['...Participant ' num2str(p) ' done.']);
            
            %lh-side
            lhfilename = [rootCodeOutputPath version '/' experimentName, '/3-single-trial-source-data/vert2562-smooth5-depth-corFM-snr1-signed/'  char(participentIDlist(p)) '-' thisword '-lh.stc'];
            if exist(lhfilename,'file')
                sensordatalh = mne_read_stc_file(lhfilename);
                
                templh(:,:,p) = sensordatalh.data(:, :);
            end
            
            %rh-side
            rhfilename = [rootCodeOutputPath version '/' experimentName, '/3-single-trial-source-data/vert2562-smooth5-depth-corFM-snr1-signed/' char(participentIDlist(p)) '-' thisword '-rh.stc'];
            if exist(rhfilename,'file')
                
                sensordatarh = mne_read_stc_file(rhfilename);
                
                temprh(:,:,p) = sensordatarh.data(:, :);
            end
        end
        
        %lh-side
        sensordatalh.data = mean(templh, 3);
        %sensordatalh.data(sensordatalh.data > 0) = 0;
        outputlh = [rootCodeOutputPath version '/' experimentName, '/4-averaged-by-trial-data/vert2562-smooth5-depth-corFM-snr1-signed/' thisword '-lh.stc'];
        mne_write_stc_file(outputlh, sensordatalh);
        
        %rh-side
        sensordatarh.data = mean(temprh, 3);
        %sensordatarh.data(sensordatarh.data > 0) = 0;
        outputrh = [rootCodeOutputPath version '/' experimentName, '/4-averaged-by-trial-data/vert2562-smooth5-depth-corFM-snr1-signed/' thisword '-rh.stc'];
        mne_write_stc_file(outputrh, sensordatarh);
        
        disp(['Average complete.']);
    end
end


