

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% make evoked sensor data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for p = 1:numel(participentIDlist) 
           
    for w = 1:numel(itemlist)
        
        thisword = char(itemlist(w));
           
            unixCommand = ['mne_process_raw '];
            
            for s = 1:participentNumBlockHash.get(char(participentIDlist(p)))
                %unixCommand = [unixCommand '--raw ' rootCodeOutputPath, version '/' experimentName, '/2-do-ICA/', char(participentIDlist(p)) '_nkg_part'  num2str(s)  '_raw_sss_movecomp_ica.fif '];
                unixCommand = [unixCommand '--raw ' rootCodeOutputPath, version '/' experimentName, '/1-preprosessing/sss/', char(participentIDlist(p)) '_nkg_part'  num2str(s)  '_raw_sss_movecomp_EEGmainline.fif '];
            end
            
            unixCommand = [unixCommand '--lowpass ' num2str(temporal_downsampling_rate) ' ' ];
            
            for s = 1:participentNumBlockHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--events ', rootCodeOutputPath, version '/' experimentName, '/3-sensor-data/eventFiles/', char(participentIDlist(p)) '_part' num2str(s) '.eve '];
            end
                
            for s = 1:participentNumBlockHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--ave  ', rootCodeOutputPath, version '/' experimentName, '/3-sensor-data/aveFiles/', char(participentIDlist(p)) '-' thisword '.ave '];
            end
               
            unixCommand = [unixCommand '--gave  ', rootCodeOutputPath, version '/' experimentName, '/3-sensor-data/fif-out-averaged/', char(participentIDlist(p)) '-' thisword '.fif '];
            
            unixCommand = [unixCommand '--projoff'];
            fprintf(['[unix:] ' unixCommand '\n']);
            unix(unixCommand);
        
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% make covarience files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for p = 1:numel(participentIDlist)

            unixCommand = ['mne_process_raw '];
            
            for s = 1:participentNumBlockHash.get(char(participentIDlist(p)))
               unixCommand = [unixCommand '--raw ' rootCodeOutputPath, version '/' experimentName, '/2-do-ICA/', char(participentIDlist(p)) '_nkg_part'  num2str(s)  '_raw_sss_movecomp_ica.fif '];
            end
            
            unixCommand = [unixCommand '--lowpass ' num2str(temporal_downsampling_rate) ' ' ];
            
            for s = 1:participentNumBlockHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--events ', rootCodeOutputPath, version '/' experimentName, '/3-sensor-data/eventFiles/', char(participentIDlist(p)) '_part' num2str(s) '.eve '];
            end

            
            for s = 1:participentNumBlockHash.get(char(participentIDlist(p)))
                unixCommand = [unixCommand '--cov ', rootCodeOutputPath, version '/' experimentName, '/3-sensor-data/cov-description-files/', char(participentIDlist(p)) '_part' num2str(s) '.cov '];
            end
            
            unixCommand = [unixCommand '--gcov ', rootCodeOutputPath, version '/' experimentName, '/3-sensor-data/averaged-noise-covarience-files/', char(participentIDlist(p)) '_gcov.fif '];

            unixCommand = [unixCommand '--projoff'];
            fprintf(['[unix:] ' unixCommand '\n']);
            unix(unixCommand);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% make plots of covarience files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Will read MNE covariance matrix files (*.fif)
% and visualised them separately for magnetometers, gradiometers and EEG
% EEG will be average-referenced
% Olaf Hauk, Nov 2010

for p = 1:numel(participentIDlist) 

  filein = [ rootCodeOutputPath, version '/' experimentName, '/3-sensor-data/averaged-noise-covarience-files/', char(participentIDlist(p)), '_gcov.fif'];    % input filename of covariance matrix file
  fprintf(1, '\nReading covariance matrix from %s\n', filein);

  covmat = mne_read_noise_cov( filein )    % read noise covariance matrix using MNE Matlab tool

  channames = covmat.names; % names of channels
  indices_EEG = strmatch('EEG', channames); % indices for EEG channels
  indices_MEG = strmatch('MEG', channames); % indices for MEG channels (grads+mags)
  for i=1:length(indices_MEG),
      lastnum(i) = str2num( channames{indices_MEG(i)}(end) );   % get last number of MEG channel names
  end;  % i

  indices_mags = indices_MEG( find ( lastnum == 1 ) );  % find magnetometer indices
  indices_grads = indices_MEG( find ( (lastnum==2)+(lastnum==3) ) );   % find gradiometer indices

  fprintf(1, 'There are %d magnetometers and %d gradiometers.\n', length(indices_mags), length(indices_grads));

  h = figure;   % create new figure

  if ~isempty(indices_EEG), % if file contains EEG...
    nr_EEG = length(indices_EEG); % number of electrodes
    fprintf(1, '...oh, and %d EEG electrodes.\n Average referencing EEG.\n\n', nr_EEG);
    covmatEEG = covmat.data(indices_EEG, indices_EEG);  % separate EEG covariance matrix
    avgop = eye(nr_EEG) - ones(nr_EEG)/nr_EEG;  % average reference operator
    covmatEEG = avgop*covmatEEG*avgop;  % apply average reference to EEG covariance matrix
    nr_plots = 3;   % plot mags, grads and EEG
    subplot(1,nr_plots,1);
    imagesc( covmatEEG );
    axis( 'square' );
    colorbar;
    th = title(['EEG '  char(participentIDlist(p))]); set(th, 'Interpreter', 'none');
  else,
      nr_plots = 2;
  end;  % plot mags and grads (no EEG)

  covmatmag = covmat.data(indices_mags, indices_mags);  % separate mags covariance matrix
  subplot(1,nr_plots,2);
  imagesc( covmatmag );
  axis( 'square' );
  colorbar;
  th = title(['Mags '  char(participentIDlist(p))]); set(th, 'Interpreter', 'none');

  covmatgrad = covmat.data(indices_grads, indices_grads);  % separate grads covariance matrix
  subplot(1,nr_plots,3);
  imagesc( covmatgrad );
  axis( 'square' );
  colorbar;
  th = title(['Grads '  char(participentIDlist(p))]); set(th, 'Interpreter', 'none');
  
  saveas(h, [rootCodeOutputPath, version '/' experimentName, '/3-sensor-data/noise_covarience_plots/' char(participentIDlist(p)) '_gcov_plot' ], 'jpg');

  close(h);
  
end