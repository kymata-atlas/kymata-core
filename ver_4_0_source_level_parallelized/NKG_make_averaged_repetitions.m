% NKG_make_averaged_repetitions.m
% Average all the repetitions between participants. Less noisy to do this
% in sensor space?

for p = 1:numel(participentIDlist)

    for w = 1:numel(wordlist)
      
        thisword = char(wordlist(w));
        
        if  ~exist([rootCodeOutputPath, 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/', char(participentIDlist(p)) '-' thisword '.fif'], 'file')
      

            if (exist([rootCodeOutputPath, 'Version4_source_level_CUDA/', experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)), '-', thisword, '-rep1.fif'], 'file') && exist([rootCodeOutputPath, 'Version4_source_level_CUDA/', experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)), '-', thisword, '-rep2.fif'], 'file'))

                sensordata2 = fiff_read_evoked([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)), '-', thisword, '-rep2.fif']);
                sensordata1 = fiff_read_evoked([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)), '-', thisword, '-rep1.fif']);

                % Average
                sensordataveraged = sensordata1;
                temp(:,:,1) = sensordata1.evoked.epochs(:, :);
                temp(:,:,2) = sensordata2.evoked.epochs(:, :);
                sensordataveraged.evoked.epochs = mean(temp, 3);

                % Save
                fiff_write_evoked([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/', char(participentIDlist(p)), '-', thisword, '.fif'], sensordataveraged);

            elseif exist([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)), '-', thisword, '-rep1.fif'], 'file')

                outputfile = ([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/', char(participentIDlist(p)), '-', thisword, '.fif']);
                copyfile([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)), '-', thisword, '-rep1.fif'], outputfile);


            elseif exist([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)), '-', thisword, '-rep2.fif'], 'file')

                outputfile = ([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/', char(participentIDlist(p)), '-', thisword, '.fif']);
                copyfile([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out/', char(participentIDlist(p)), '-', thisword, '-rep2.fif'], outputfile);


            else
                    % Nothing
            end
            clear temp;
        end 
     end
end
