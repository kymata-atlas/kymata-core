
%%% Remove eyebinks from sensor data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath /imaging/local/spm_eeglab/
addpath /imaging/local/eeglab/
addpath /imaging/at03/NKG_Code/Version4_2/ICA_functions/

%%% STEP ONE
%%% CW & YH - 08/12
%%% RUN IN SPM ($ spm 5 matlab2009a eeg l63)
%%%%%%%%%%%%%%%Convert to SPM%%%%%%%%%%%%%%%


origdatapath = '/imaging/ef02/phrasal';


for p = 1:numel(participentIDlist)
   
    for s = 1:participentSessionHash.get(char(participentIDlist(p)))
        
        fname = [origdatapath, '/', char(participentIDlist(p)), '/tr/' char(participentIDlist(p)), '_part'  num2str(s),  '_raw_sss_movecomp_tr.fif'];
        
        [fpath fstem ext] = fileparts(fname);
        
        %%% IMPORT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        clear S D
        S.Fdata      = fname;
        S.Pout       = [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', fstem, '-SPM.mat'];
        S.Fchannels  = '/imaging/local/spm/spm5/EEGtemplates/FIF306_setup.mat';
        S.Fchannels_eeg = fullfile(spm('dir'),'EEGtemplates','cbu_meg_70eeg_montage.mat');
        S.eeg_ref    = 'average';      % Function will re-reference to average of EEG data
        S.twin       = [0 Inf];
        S.trig_chan  = 'STI101';
        S.veogchan   = [62];
        S.heogchan   = [61];
        S.ecgchan    = [0];
        S.veog_unipolar = 0;
        S.heog_unipolar = 0;
        S.dig_method = 1;           % new digitisation
        S.conds      = 0;
        S.HPIfile    = [origdatapath, '/', char(participentIDlist(p)) '/raw/' char(participentIDlist(p)) '_part' num2str(s) '_raw.fif']; 
        S.grms       = 0;
        D            = spm_eeg_rdata_FIF(S);
        
        %%% Split Grads and Mags (need to be on 64-bit SPM) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        clear S D
          
        S.Fdata      = fname;
        S.D          = [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', fstem, '-SPM.mat'];       % structure for splitting
        D            = spm_eeg_splitFIF(S);   % split SPM files
         
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Mark bad EEG channels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% D.channels.Bad needs to contain all 70 EEG with 0s in place of good EEG and
% bad EEG marked with their number in order (e.g. 7 as the 7th number).


for p = 1:numel(participentIDlist)
    
    
    clear S D
    badchanfilename = [origdatapath, '/' char(participentIDlist(p)) '/tr/bad.txt'];
    fid = fopen(badchanfilename);
    
    if fid == 3
        badchannels = textscan(fid, '%s %d', 'delimiter', ' ');
        %if no file, or empty then no bad channels
    
        % As the EEG in D.channels.eeg doesn't skip from 60-65, for the channels
        % 65-74, they should be 61-70 instead.
        for i = 1:length(badchannels{1,2})
            if (badchannels{1,2}(i) >= 65 && strcmp(badchannels{1,1}(i),'EEG'))
                badchannels{1,2}(i) = badchannels{1,2}(i)-4;
            end
        end
        
    else
        badchannels{1,2} = [];
    end
    fclose('all');
  
    for s = 1:participentSessionHash.get(char(participentIDlist(p)))

        fstem = [char(participentIDlist(p)), '_part'  num2str(s),  '_raw_sss_movecomp_tr'];

        load ([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', fstem, '-SPM-eeg.mat']);
        D.channels.Bad = zeros(1,70);

        for  i = 1:length(badchannels{1,2})
            D.channels.Bad(badchannels{1,2}(i)) = badchannels{1,2}(i);
        end
        
        save([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', fstem, '-SPM-eeg-withbadchannelsmarked.mat'], 'D');

    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% RUN ICA - outputs an ica_*.dat file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/TEST/']);

for p = 1:numel(participentIDlist)
    
    for s = 1:participentSessionHash.get(char(participentIDlist(p)))
    
        
        fname = [rootCodeOutputPath, version '/', char(participentIDlist(p)), '/tr/' char(participentIDlist(p)), '_part'  num2str(s),  '_raw_sss_movecomp_tr.fif'];
        [remove fstem ext] = fileparts(fname);
        
        % Mags
        clear S D
        S.D       = [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/TEST/', fstem, '-SPM-mags.mat'];
        S.samppct = 1;
        S.excbad  = 0;
        S.newpath = [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/TEST/'];
        S.args    = {'extended',1,'maxsteps',800,'pca',32};
        D = spm_eeglab_runica_yh(S);

        % Grads
        clear S D
        S.D       = [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', fstem, '-SPM-grds.mat'];
        S.samppct = 1;
        S.excbad  = 0;
        S.newpath = [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/'];
        S.args    = {'extended',1,'maxsteps',800,'pca',32};
        D = spm_eeglab_runica_yh(S);

        % EEG marked with bad channels
        clear S D
        S.D       = [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/TEST/', fstem, '-SPM-eeg-withbadchannelsmarked.mat'];
        S.samppct = 1;
        S.excbad  = 1;
        S.newpath = [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/TEST/'];
        S.args    = {'extended',1,'maxsteps',800,'pca',32};
        D=spm_eeglab_runica_yh(S);
        
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--Classify and Remove Artefact ICs-------------------------
%
% View raw data on blink- and pulse-sensitive channels,
%  compute IC activations, correlate with physiological 
%  signals, plot corr Z-scores, plot topographies,
%  classify ICs as blink, horizontal eye mov'ts, pulse,
%  and remove selected artefact components.

clear S D

for p = 1:numel(participentIDlist);
    for s = 1:participentSessionHash.get(char(participentIDlist(p)))

        rawname = ['/imaging/ef02/phrasal/', char(participentIDlist(p)), '/tr/' char(participentIDlist(p)), '_part'  num2str(s),  '_raw_sss_movecomp_tr.fif'];
        [remove fstem ext] = fileparts(rawname);

        ica_class    = {'blink','horiz'};   % Artefact(s) to classify
        ica_chans    = {'veog','heog'};     % Reference channels for correlations
        ica_rem      = {'blink','horiz'};   % Artefact(s) to remove
       
        %eeg
        
        icafname=sprintf('ica_%s-SPM-eeg.mat',fstem);

        D.fname=icafname;

        clear S
        S.D = D.fname;
        S.mode     = {'both'};  % classify + remove
        S.artlabs  = ica_class;
        S.chanlabs = ica_chans;
        S.remlabs  = ica_rem;
        S.remfname = ['rem_' S.D];

        D = meg_ica_artefact_eeg(S);

        ica_class    = {'blink'};   % Artefact(s) to classify - (left out horiz as (most of the time) not in ICA components for MEG )
        ica_chans    = {'veog'};     % Reference channels for correlations
        ica_rem      = {'blink'};   % Artefact(s) to remove
        
        %Grads

        icafname=sprintf('ica_%s-SPM-grds.mat',fstem);

        D.fname=icafname;
        
        clear S
        S.D = D.fname;
        S.mode     = {'both'};  % classify + remove
        S.artlabs  = ica_class;
        S.chanlabs = ica_chans;
        S.remlabs  = ica_rem;
        S.remfname = ['rem_' S.D];

        D = meg_ica_artefact_grds(S);
        
        % Mags
        
        icafname=sprintf('ica_%s-SPM-mags.mat',fstem);

        D.fname=icafname;

        clear S
        S.D = D.fname;
        S.mode     = {'both'};  % classify + remove
        S.artlabs  = ica_class;
        S.chanlabs = ica_chans;
        S.remlabs  = ica_rem;
        S.remfname = ['rem_' S.D];

        D = meg_ica_artefact_mags(S);

    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Before recombining grads, mags and EEG into fif, the EEG has to have 70
%%% channels. In the rem_ICA_*-eeg files, these bad channels are missing, so this script
%%% will insert the badchannels from the original EEG data. In fact, it takes the origianl 
%%% EEG data, replaces the old data with ICA data, and saves it with a new name.
%%%
%%% CW July 2012


clear S D

for p = 1:numel(participentIDlist)
    for s = 1:participentSessionHash.get(char(participentIDlist(p)))

        rawname = ['/imaging/ef02/phrasal/', char(participentIDlist(p)), '/tr/' char(participentIDlist(p)), '_part'  num2str(s),  '_raw_sss_movecomp_tr.fif'];
        [remove fstem ext] = fileparts(rawname);

        rem_ica_fname=sprintf('rem_ica_%s-SPM-eeg.mat',fstem);
        orig_fname=sprintf('%s-SPM-eeg.mat',fstem);

        blinksremoveddata  = spm_eeg_ldata([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', rem_ica_fname ]);
        origdata            = spm_eeg_ldata([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', orig_fname ]);
        
%          %temp - this was for participent 2 in phrasal data who had an extra
%          % channel (63) that they shouldnt have had
%           
%            A1 = origdata.data(1:60,:);
%            A2 = origdata.data(62:end,:);
%            origdata.data = [A1 ; A2];
%            origdata.Nchannels = 72; 
%            origdata.channels.reference(:,61) = []; 
%            origdata.channels.order(:,61) = []; 
%            origdata.channels.eeg(61) = [];
%            origdata.channels.scaled(61,:) = []; 
%            origdata.scale(61,:) = [];
%            origdata.channels.heog = 72; 
%            origdata.channels.veog = 71;
%            origdata.channels.name(61) = [];
%            
%            A1 = blinksremoveddata.data(1:59,:);
%            A2 = blinksremoveddata.data(61:end,:);
%            blinksremoveddata.data = [A1 ; A2]; %
%          
             
        tempdata = zeros(70,length(blinksremoveddata.data));
        k=0;
        for channel=1:length(blinksremoveddata.channels.Bad)
            if blinksremoveddata.channels.Bad(1,channel)~=0
                tempdata(channel,:)=origdata.data(channel,:);
                k=k+1;
            elseif blinksremoveddata.channels.Bad(1,channel) == 0
                tempdata(channel,:)=blinksremoveddata.data(channel-k,:);
            end
        end
        tempdata(71,:) = origdata.data(71,:);
        tempdata(72,:) = origdata.data(72,:);
        
        rem_ica_allchans_fname=sprintf('rem_ica_%s-allchannels-SPM-eeg.mat',fstem);
        rem_ica_allchans_datfname=sprintf('rem_ica_%s-allchannels-SPM-eeg.dat',fstem);
        rem_ica_allchans_fname
        
        % The .dat file is re-written by mablab only if certain syntax is
        % used. This is the attempt to make it work (could possibly be cleaned up.)
        
        eval(['! cp ' [rootCodeOutputPath, version '/' experimentName,'/1-do-ICA/1-outputSPMfiles/' fstem '-SPM-eeg.dat' ] ' ' [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/' rem_ica_allchans_datfname]]);
        eval(['! cp ' [rootCodeOutputPath, version '/' experimentName,'/1-do-ICA/1-outputSPMfiles/' fstem '-SPM-eeg.mat' ] ' ' [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/' rem_ica_allchans_fname]]);
        D = spm_eeg_ldata([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', rem_ica_allchans_fname ]);
        D.fname = rem_ica_allchans_fname;
        D.fnamedat = rem_ica_allchans_datfname;
        save([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/' rem_ica_allchans_fname], 'D');
        clear D
        D = spm_eeg_ldata([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', rem_ica_allchans_fname ]);
        D.fname = rem_ica_allchans_fname;
        D.fnamedat = rem_ica_allchans_datfname;
        D.data(1:72,:) = tempdata(:,:);
        D.ica = blinksremoveddata.ica;
        save([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/' rem_ica_allchans_fname], 'D');
        clear D

        %test
        %D = spm_eeg_ldata([rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/1-outputSPMfiles/', rem_ica_allchans_fname ]);
        %D.fname
        %D.fnamedat
        %ExampleD = D.data(:,1:1000);
        %Exampletempdata = tempdata(:,1:1000);

    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% convert SPM EEG/Mag/Grad data back to .fif----------------------------

clear all

addpath /imaging/ef02/ICA_recomb_script/
addpath /imaging/local/software/mne/mne_2.7.0/x86_64/mne/share/matlab/
addpath /usr/local/matlab/r2009a/toolbox/
          
for p = 2:numel(participentIDlist);
    
    for s = 1:participentSessionHash.get(char(participentIDlist(p)))
        
        clear D_mag D_eeg D_grad data_or D_mag_temp D_grad_temp D_eeg_temp
        
        fprintf(['Working on subject: [' char(participentIDlist(p)),'] and part [' num2str(s) ']\n']);

        rawfile   = ['/imaging/ef02/phrasal/', char(participentIDlist(p)), '/tr/' char(participentIDlist(p)), '_part'  num2str(s),  '_raw_sss_movecomp_tr.fif'];
        [remove fstem ext] = fileparts(rawfile);

        GradFile  = ['rem_ica_' fstem '-SPM-grds.mat'];
        MagFile   = ['rem_ica_' fstem '-SPM-mags.mat'];
        EEGFile   = ['rem_ica_' fstem '-allchannels-SPM-eeg.mat'];

        outFile   = [rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/2-outputfiffiles/ICA_',char(participentIDlist(p)),'_part', num2str(s), '_sss_movecomp_tr.fif'];

        data_or = fiff_setup_read_raw(rawfile);
        A = data_or.last_samp-data_or.first_samp+1;
        new_data = zeros(379,A);

        D_grad = spm_eeg_ldata(GradFile); % work on Grads
        D_grad_temp = D_grad.data(1:206,:)*1e-15;

        t=1;
        for l=1:2:204
            if l==1;
                new_data(l:1+1,:) = D_grad_temp(l:l+1,:);
                t=t+1;
            else
                new_data([t*3-2:t*3-1],:) = D_grad_temp(l:l+1,:);
                t=t+1;
            end % if
        end % for l

        clear l t

        D_mag = spm_eeg_ldata(MagFile); % work on Mag
        D_mag_temp = D_mag.data(1:102,:)*1e-15;

        for l=1:102
            new_data(l*3,:) = D_mag_temp(l,:);
        end % for l

        clear l

        D_eeg = spm_eeg_ldata(EEGFile); % work on EEG
        D_eeg_temp = D_eeg.data(1:72,:)*1e-6;

        new_data(307:366,:)= D_eeg_temp(1:60,:);
        new_data(367:376,:)=D_eeg_temp(61:70,:);
        new_data(379,:)=D_eeg_temp(71,:); % VEOG
        new_data(378,:)=D_eeg_temp(72,:); %HEOG

        clear D_mag D_eeg D_grad data_or D_mag_temp D_grad_temp D_eeg_temp

        mne_read_write_raw_NKG(rawfile,outFile,new_data, ['/imaging/ef02/phrasal/',  char(participentIDlist(p)) , '/raw/',  char(participentIDlist(p)), '_part1_avg.fif'])

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% fix eeg positions

Run fix eeg!


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% mark bad channels in .fif file. you will need to do this bit in mne
%% environment


for p = 1:numel(participentIDlist)

    for s = 1:participentSessionHash.get(char(participentIDlist(p)))

        if exist(['/imaging/ef02/phrasal/' char(participentIDlist(p)) '/tr/bad.txt'], 'file')

            %eval(['! cp ' ['/imaging/ef02/phrasal/' char(participentIDlist(p)) '/tr/bad.txt'] ' ' ['/home/at03/Desktop/' char(participentIDlist(p)) '_bad.txt']]);

            unixCommand   = ['mne_mark_bad_channels '];
            unixCommand   = [unixCommand ' --bad  /home/at03/Desktop/verbphrase_bad_channels/' char(participentIDlist(p)) '_bad.txt '];
            %unixCommand   = [unixCommand ' --bad  /imaging/ef02/phrasal/' char(participentIDlist(p)) '/tr/bad.txt '];
            unixCommand   = [unixCommand rootCodeOutputPath, version '/' experimentName, '/1-do-ICA/2-outputfiffiles/ICA_',char(participentIDlist(p)),'_part', num2str(s), '_sss_movecomp_tr.fif'];
            unix(unixCommand);

        end

    end
end

