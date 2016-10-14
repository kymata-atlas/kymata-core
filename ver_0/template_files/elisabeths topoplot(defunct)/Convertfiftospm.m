
addpath /opt/mne/matlab/toolbox
addpath /imaging/local/spm/spm5
addpath /neuro/meg_pd_1.2
addpath /imaging/local/spm/spm5/EEGtemplates
addpath /imaging/local/linux/mne_2.6.0/mne/matlab/toolbox
addpath /imaging/local/spm/spm5/cbu_updates

% Modified from http://imaging.mrc-cbu.cam.ac.uk/meg/SensorSPM
% takes a list of Fiff-files (cell array "fifflist{}") and converts MEG data to SPM image files,
% which can then be subjected to 2nd-level statistics, for example
% "rootdir" can contain the directory path common to all files
% the script will produce subdirectories with img-files in the directories of the input fiff-files
% so far only tested on one data set
% OH, March 2009
% EF, June 09 modified for ERP criteria

clear all

% you can specify a list of fiff-files here, to be converted into SPM format

fifflist = {'test.fif'};

nr_fiffs = length(fifflist);

% root directory common to all input fiff-files (not required if full
% pathnames specified in fifflist{})
root_dir = '/imaging/at03/Fieldtrip_recogniser_coherence/convert_to_topoplot/';
out_dir = '/imaging/at03/Fieldtrip_recogniser_coherence/convert_to_topoplot/out';

% GET GOING...
clear S;
for ff = 1:nr_fiffs,    % for all Fiff-files...
% CONVERT FROM FIFF TO SPM FORMAT...
    S.Fchannels = fullfile(spm('dir'),'EEGtemplates','FIF306_setup.mat');
    S.Fchannels_eeg = fullfile(spm('dir'),'EEGtemplates','cbu_meg_70eeg_ecg_montage.mat');
    S.veogchan  = [62];         % Channel 62 is (bipolar) VEOG, MAY NEED EDITING
    S.heogchan  = [61];         % Channel 61 is (bipolar) HEOG, MAY NEED EDITING
    S.ecgchan = [63 64];        % EEG channel number(s) for ECG channel
    S.ecg_unipolar = 0;         %  whether to subtract two ECG channels ([1=yes,0=no])
    S.veog_unipolar = 0;        % - whether to subtract two VEOG channels ([1=yes,0=no])
    S.heog_unipolar = 0;        % - whether to subtract two HEOG channels ([1=yes,0=no])
    S.dig_method = 0;           % - whether old or new CBU lab method for recording+digitising EEG (http://imaging.mrc-cbu.cam.ac.uk/meg/CbuSpmParameters)
    S.conds = 1;                % conditions in fiff-file, MAY NEED EDITING
    S.eeg_ref = 'average';      % Function will re-reference to average of EEG data
    S.grms = 1;                 % so that the splitting outputs mags, grads and grad RMS (grms)
    [fiffpath,fiffname,fiffext,fiffversn] = fileparts(fifflist{ff});
    S.Fdata = fullfile(root_dir, fifflist{ff});                      % Input Fiff-file before splitting
    fileSPMout = fullfile(out_dir, fiffpath, [fiffname '_SPM.mat']); % Output SPM file (if not specified, will be 'myexp.mat')
    S.Pout  = fileSPMout;                                            % output for conversion from Fiff to SPM format (before splitting)
    D0 = spm_eeg_rdata_FIF(S);                                       % convert fiff- to SPM-format

% SPLIT DATA INTO MAGS/GRADS/EEG...
    S.D = fileSPMout;           % structure for splitting
    D1 = spm_eeg_splitFIF(S);   % split SPM files

% WRITE TO IMAGE VOLUMES:
    % Select options (see help for spm_eeg_convertmat2ana3D):
    clear S
    S.interpolate_bad = 0;
    S.n = 32;
    S.pixsize = 3;
    % Select trial types:
    S.trialtypes = [1];
    img_outnames = '';
    % Mags:
    S.Fname = fullfile(out_dir, fiffpath, [fiffname '_SPM-mags.mat']);
    tmpname = spm_eeg_convertmat2ana3D(S);
    img_outnames(1,:) = fullfile(out_dir, fiffpath, [fiffname '_SPM-mags'], 'trialtype1', 'average.img');
    % Grad magnitude:
    S.Fname = fullfile(out_dir, fiffpath, [fiffname '_SPM-grds.mat']);
    tmpname = spm_eeg_convertmat2ana3D(S);
    img_outnames(2,:) = fullfile(out_dir, fiffpath, [fiffname '_SPM-grds'], 'trialtype1', 'average.img');
    % GradRMS magnitude:
    S.Fname = fullfile(out_dir, fiffpath, [fiffname '_SPM-grms.mat']);
    tmpname = spm_eeg_convertmat2ana3D(S);
    img_outnames(3,:) = fullfile(out_dir, fiffpath, [fiffname '_SPM-grms'], 'trialtype1', 'average.img');
    
% SMOOTH IMAGES
    P = img_outnames;
    SmoothKernel = [5 5 10];        % Smoothness in x (mm), y (mm) and z (ms), MAY NEED EDITING
    for n=1:size(P,1);
               [pth,nam,ext] = fileparts(P(n,:));
               Pout{n}       = fullfile(pth,['s' nam ext]);
               spm_smooth(spm_vol(P(n,:)),Pout{n},SmoothKernel);
               Pin = strvcat(P(n,:),Pout{n});
               spm_imcalc_ui(Pin,Pout{n},'((i1+eps).*i2)./(i1+eps)',{[],[],'float32',0});
    end
    
% ERPs magnitude:
    S.Fname = fullfile(out_dir, fiffpath, [fiffname '_SPM-eeg.mat']);
    tmpname = spm_eeg_convertmat2ana3D(S);
    img_outnames_eeg = fullfile(out_dir, fiffpath, [fiffname '_SPM-eeg'], 'trialtype1', 'average.img');

% SMOOTH IMAGES
    P = img_outnames_eeg;
    SmoothKernel = [5 5 10];        % Smoothness in x (mm), y (mm) and z (ms), MAY NEED EDITING
    for n=1:size(P,1);
               [pth,nam,ext] = fileparts(P(n,:));
               Pout{n}       = fullfile(pth,['s' nam ext]);
               spm_smooth(spm_vol(P(n,:)),Pout{n},SmoothKernel);
               Pin = strvcat(P(n,:),Pout{n});
               spm_imcalc_ui(Pin,Pout{n},'((i1+eps).*i2)./(i1+eps)',{[],[],'float32',0});
    end
        
    %The final imcalc step above is just to reinsert NaN's for voxels outside space-time volume into which data were smoothed
end;    %..ff

