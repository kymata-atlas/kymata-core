
% unsing elisabeth's inverse solution

for p = 1:numel(participentIDlist) 
    
    for w = 1:numel(wordlist) 
        
        thisword = char(wordlist(w));
        
        if exist(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-averaged/meg08_0' num2str(participentIDlist(p)) '-' thisword '.fif'], 'file')
        
            unixCommand = ['mne_make_movie '];
            unixCommand = [unixCommand '--inv /imaging/ef02/lexpro/meg08_0' num2str(participentIDlist(p)) '/tr/old_meg08_0' num2str(participentIDlist(p)) '_3L-loose0.2-depth-reg-inv.fif ' ];
            unixCommand = [unixCommand '--meas /imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-averaged/meg08_0' num2str(participentIDlist(p)) '-' thisword '.fif ' ];
            unixCommand = [unixCommand '--morph average ' ];
            unixCommand = [unixCommand '--morphgrade 3 ' ]; % downsamples to 642 is 3
            unixCommand = [unixCommand '--subject 0' num2str(participentIDlist(p)) ' ' ];
            unixCommand = [unixCommand '--stc /imaging/at03/NKG/saved_data/source_space/3-averaged-by-trial-data/spatially_downsampled/elisabeth_invsol/0' num2str(participentIDlist(p)) '-' thisword ' '];
            unixCommand = [unixCommand '--smooth 5 ' ];
            fprintf(['[unix:] ' unixCommand '\n']);
            unix(unixCommand);
            
        end
    end
end




%olafs method


for p = 1:1%numel(participentIDlist)
    
    for w = 1:1%numel(wordlist)
        
        thisword = char(wordlist(w));
        
        if exist(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-averaged/meg08_0' num2str(participentIDlist(p)) '-' thisword '.fif'], 'file')

            %invorig = mne_read_inverse_operator(['/imaging/at03/LexproMEG/meg08_0' num2str(participentIDlist(p)) '/inv/meg08_0' num2str(participentIDlist(p)) '_3L-loose0.2-depth-reg-inv.fif']);

            %inv = mne_get_inverse_matrix(['/imaging/at03/LexproMEG/meg08_0' num2str(participentIDlist(p)) '/inv/meg08_0' num2str(participentIDlist(p)) '_3L-loose0.2-depth-reg-inv.fif'],1,SNR);     

            
            
            invorig = mne_read_inverse_operator(['/imaging/ef02/lexpro/meg08_0' num2str(participentIDlist(p)) '/tr/meg08_0' num2str(participentIDlist(p)) '_3L-loose0.2-depth-reg-inv.fif']); % change to mine      

            inv = mne_get_inverse_matrix(['/imaging/ef02/lexpro/meg08_0' num2str(participentIDlist(p)) '/tr/meg08_0' num2str(participentIDlist(p)) '_3L-loose0.2-depth-reg-inv.fif'],1,SNR); % change to mine      
            sensordata = fiff_read_evoked(['/imaging/at03/NKG/saved_data/source_space/1-sensor-data/fif-out-averaged/meg08_0' num2str(participentIDlist(p)) '-' thisword '.fif']);
            
            badchannels = inv.inv.noise_cov.bads;
            
            listofgoods =ones(1, numel(sensordata.info.ch_names));
            for m = 1: numel(inv.inv.noise_cov.bads)
                for i = 1:numel(sensordata.info.ch_names)
                    if strcmp(sensordata.info.ch_names(i),badchannels(m))
                        listofgoods(1, i) = 0;
                    elseif numel(strfind(sensordata.info.ch_names{i}, 'STI')) > 0
                        listofgoods(1, i) = 0;
                    elseif numel(strfind(sensordata.info.ch_names{i}, 'MISC')) > 0
                        listofgoods(1, i) = 0;
                    elseif numel(strfind(sensordata.info.ch_names{i}, 'CHP')) > 0
                        listofgoods(1, i) = 0;
                    elseif numel(strfind(sensordata.info.ch_names{i}, 'EOG')) > 0
                        listofgoods(1, i) = 0;
                    elseif numel(strfind(sensordata.info.ch_names{i}, 'ECG')) > 0
                        listofgoods(1, i) = 0;
                    end
                end
            end
            
            gooddata = [];
            j=1;
            for i = 1:length(listofgoods)
                if listofgoods(1, i) == 1
                    gooddata(j, :) = sensordata.evoked.epochs(i, :);
                    j = j+1;
                end
            end
            
            % get inverse solution
            locations = gooddata'*inv.invmat';
            
            % get intensity (only for loose, comment out if 'fixed orientation')
            intensity = zeros(size(locations, 1), size(locations, 2)/3);
            for j = 1:size(locations, 1)
                for i = 1:3:length(locations)
                    intensity (j, (i+2)/3) = sqrt(locations(j,i)^2 + locations(j,i+1)^2 + locations(j,i+2)^2);
                end
            end
           
                
            mne_write_inverse_sol_stc(['/imaging/at03/NKG/saved_data/source_space/2-single-trial-source-data/meg08_0' num2str(participentIDlist(p)) '-' thisword ], invorig, intensity',  -0.001*pre_stimulus_window, 0.001)
                  
            clear ourSourceSpace source gooddata inv sensordata  invorig
            
        end
        
    end
    
end
