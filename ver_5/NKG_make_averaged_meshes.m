

cbupool(9) % any more and it is likely to run out of memory with the default options.

% Average STCs
parfor w = 1:numel(itemlist)
    
    thisitem = char(itemlist(w));
    
    if ~exist([rootCodeOutputPath version '/' experimentName, '/5-averaged-by-trial-data/vert10242-nodepth-diagonly-snr1-signed-baselineNone/visual/' thisitem '-rh.stc'],'file')
        
        disp([thisitem]);
        temprh = [];
        templh = [];
        sensordatarh = [];
        sensordatalh = [];
        
        for p = 1:numel(participentIDlist)
            
            disp(['...Participant ' num2str(p) ' done.']);
            
            %lh-side
            lhfilename = [rootCodeOutputPath version '/' experimentName, '/4-single-trial-source-data/vert10242-nodepth-diagonly-snr1-signed-baselineNone/visual/'  char(participentIDlist(p)) '-' thisitem '-lh.stc'];
            if exist(lhfilename,'file')
                sensordatalh = mne_read_stc_file(lhfilename);
                
                templh(:,:,p) = sensordatalh.data(:, :);
            end
            
            %rh-side
            rhfilename = [rootCodeOutputPath version '/' experimentName, '/4-single-trial-source-data/vert10242-nodepth-diagonly-snr1-signed-baselineNone/visual/' char(participentIDlist(p)) '-' thisitem '-rh.stc'];
            if exist(rhfilename,'file')
                
                sensordatarh = mne_read_stc_file(rhfilename);
                
                temprh(:,:,p) = sensordatarh.data(:, :);
            end
        end
        
        %lh-side
        sensordatalh.data = mean(templh, 3);
        %sensordatalh.data(sensordatalh.data < 0) = 0;
        outputlh = [rootCodeOutputPath version '/' experimentName, '/5-averaged-by-trial-data/vert10242-nodepth-diagonly-snr1-signed-baselineNone/visual/' thisitem '-lh.stc'];
        mne_write_stc_file(outputlh, sensordatalh);
        
        %rh-side
        sensordatarh.data = mean(temprh, 3);
        %sensordatarh.data(sensordatarh.data < 0) = 0;
        outputrh = [rootCodeOutputPath version '/' experimentName, '/5-averaged-by-trial-data/vert10242-nodepth-diagonly-snr1-signed-baselineNone/visual/' thisitem '-rh.stc'];
        mne_write_stc_file(outputrh, sensordatarh);
        
        disp(['Average complete.']);
    else
        disp(['Ignored ', thisitem]);
    end
end

matlabpool close;


