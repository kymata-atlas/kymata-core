% NKG_make_grandaverage.m

numberoffilestobeaveraged = 1865;
temp = zeros(381,length_of_longest_stimuli+post_stimulus_window+pre_stimulus_window+1, numberoffilestobeaveraged); % this last dimension is unclear because we don't know how many items are missing. Always check before adveraging.
count = 0;

%load channelnames

sensordataveraged = fiff_read_evoked([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/gave/', char(participentIDlist(1)), '-', char(wordlist(4)), '.fif']);


%average

for p = 1:numel(participentIDlist)

    for w = 1:numel(wordlist)
      
        thisword = char(wordlist(w));
        
        if (exist([rootCodeOutputPath, 'Version4_source_level_CUDA/', experimentName, '/1-sensor-data/fif-out-averaged/gave/', char(participentIDlist(p)), '-', thisword, '.fif'], 'file'))

                sensordata = fiff_read_evoked([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/gave/', char(participentIDlist(p)), '-', thisword, '.fif']);
                count = count+1;
             
                % Average
                %For each channel name
                for channel = 1:381
                    temp(channel,:,count) = sensordata.evoked.epochs(find(strcmp(sensordataveraged.info.ch_names(channel), sensordata.info.ch_names)),:);
                end

        else
                    % Nothing
        end
    end
    
end

sensordataveraged.evoked.epochs(1:381,:) = mean(temp, 3);

% Save
fiff_write_evoked([rootCodeOutputPath 'Version4_source_level_CUDA/' experimentName, '/1-sensor-data/fif-out-averaged/gave/GRANDAVERAGE.fif'], sensordataveraged);

