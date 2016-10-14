% NKG_make_averaged_repetitions.m
% Average all the repetitions between participants. Less noisy to do this
% in sensor space.

for p = 1:numel(participentIDlist)

    for w = 1:numel(wordlist)
        
        thisword = char(wordlist(w));

        if exist(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-sorted/meg08_0' num2str(participentIDlist(p)) '-' thisword '-rep2.fif'], 'file')

            sensordata2 = fiff_read_evoked(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-sorted/meg08_0' num2str(participentIDlist(p)) '-' thisword '-rep2.fif']);
            sensordata1 = fiff_read_evoked(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-sorted/meg08_0' num2str(participentIDlist(p)) '-' thisword '-rep1.fif']);

            % Average
            sensordataveraged = sensordata1;
            temp(:,:,1) = sensordata1.evoked.epochs(:, :);
            temp(:,:,2) = sensordata2.evoked.epochs(:, :);
            sensordataveraged.evoked.epochs = mean(temp, 3);
            
            % Save
            fiff_write_evoked(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-averaged/meg08_0' num2str(participentIDlist(p)) '-' thisword '.fif'], sensordataveraged);
            
        elseif exist(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-sorted/meg08_0' num2str(participentIDlist(p)) '-' thisword '-rep1.fif'], 'file')
            
            outputfile = (['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-averaged/meg08_0' num2str(participentIDlist(p)) '-' thisword '.fif']);
            copyfile(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-sorted/meg08_0' num2str(participentIDlist(p)) '-' thisword '-rep1.fif'], outputfile);
  
            
        else
                % Nothing
        end
        clear temp;
     end
end
