function [D,varargout] = spm_eeg_rdata_FIF(S)
% function to read in NeuroMag *.FIF data to SPM5
% FORMAT Do = spm_eeg_rdata_FIF(S)
% 
% S - struct; (optional) fields of S:
% Fdata     - continuous or averaged FIF file to read
% path      - new path for output
% Pout      - new filename for output
% Dtype     - data type (eg, 'int16' or 'float32')
% Fchannels - MEG channel template file
% sclchan   - [0|1] turn channel scaling on or off
% sclfact   - channel scale factors
% HPIfile   - .fif or .mat file (with full path) from which to load HPI points
% conds     - which sets to load from an averaged data file
% twin      - time window (ms) to read, relative to start of continuous
%             data, or event of averaged data
% trig_chan - trigger channel for continuous data (=0 if none)
% trig_type - type of deflection indicating onset of event (see below)
% veogchan  - EEG channel number(s) for VEOG channel
% heogchan  - EEG channel number(s) for HEOG channel
% ecgchan - EEG channel number(s) for ECG channel
% veog_unipolar    - whether to subtract two VEOG channels ([1=yes,0=no])
% heog_unipolar    - whether to subtract two HEOG channels ([1=yes,0=no])
% ecg_unipolar     - whether to subtract two ECG channels ([1=yes,0=no])
%
% ...and for any concurrent EEG:
% Fchannels_eeg - EEG channel template file
% eeg_ref       - Reference channel name (if 'average', then averaged calculated)
% dig_method    - whether old or new CBU lab method for recording+digitising
%                 EEG (http://imaging.mrc-cbu.cam.ac.uk/meg/CbuSpmParameters)
%
% EXAMPLE batch call:
%  
% S.Fdata = 'myexp.fif'       % Input file
% S.Pout  = 'myexp_spm.mat'   % Output SPM file (if not specified, will be 'myexp.mat')
% S.Fchannels = fullfile(spm('dir'),'EEGtemplates','FIF306_setup_ecg.mat');  
%               
% SPM 2D display template (must in in EEGtemplates subdir of SPM dir)
% S.veogchan  = [62];         % Channel 62 is (bipolar) VEOG
% S.heogchan  = [61];         % Channel 61 is (bipolar) HEOG
% S.ecgchan = [63];           % Channel 63 is (bipolar) ECG
% S.veog_unipolar = 0;        % As I said above, not unipolar ;-)
% S.heog_unipolar = 0;        % As I said above, not unipolar ;-)
% S.ecg_unipolar = 0;         % As I said above, not unipolar ;-)
% S.trig_chan = 'STI101';     % Trigger channels
% S.twin = [0 Inf];           % Read all samples
% If have concurrent EEG data:
% S.Fchannels_eeg = fullfile(spm('dir'),'EEGtemplates','cbu_meg_70eeg_ecg_montage.mat');   
%               
% SPM 2D display template (must in in EEGtemplates subdir of SPM dir)
% S.eeg_ref = 'average';      % Function will re-reference to average of EEG data
%
% NOTES: 
%    - Requires the Fiff Access toolbox written by Kimmo Uutela
%      http://www.kolumbus.fi/~w132276/programs/meg-pd/
%    - Fiff Access does not handle long int, so may need to convert 
%      to short int or float first (eg using MaxFilter)
%    - Does NOT apply any SSP vectors within FIF file
%    - Requires separate function spm_eeg_rdata_FIF_channels.m
%      This function returns channel information in Brainstorm format,
%      only part of which is stored in SPM's D structure.
%    - If two VEOG or HEOG channels, will subtract to calculate difference
%      (ie assumes coded as unipolar channels, not already bipolar)
%
% Will put SPM5 *.mat file in same directory as FIF file unless 
% alternative S.path or S.Pout passed
%
% Rik Henson (5/06/07), with thanks to Danny Mitchell and Jason Taylor
% RH Updated 20/11/07 to handle larger raw files
% RH Updated 8/1/08 to handle skips in raw files
% RH Updated 10/1/08 to handle concurrent EEG data (written to a separate file)
% RH Updated 17/1/08 to handle 2 VEOG or 2 HEOG channels (as happens when 
%                    use 124 EEG montage, and 61-64 (EOG) treated as EEG)
% RH Updated 19/3/08 to allow specification of different trigger
%                    definitions in raw data (beware of downsampling!)
% RH Updated 21/5/08 to read in Bad channels marked in FIF
% RH Updated 6/6/08  to handle different datatypes, and varargout{1} to
%                    return trig channel
% JT Updated 17/6/08 to import ECG channel(s); handled same as EOG
% RH Updated 5/3/09  to handle empty room data (where no HPI)
% RH Updated 13/3/09 not to scale by default, and correct grad weights in m
% DJM Updated 21/4/09 added new trig_type options of 'new_stable_change'
%                    and 'new_stable_change_recode_buttons'
% RH Updated 21/5/09 to handle old vs new EEG digitisation (see
%                    S.dig_methods above)
% JT Updated 9/7/09  only import events within time-window (and resynch time)

try
    	Fdata = S.Fdata;
catch
    	Fdata = spm_select(1, '\.fif$', 'Select FIF file');
end
P = spm_str_manip(Fdata, 'H');

try
    	D.path = S.path;
catch
    	D.path = P;
end

if exist(Fdata)==2
    if loadfif(Fdata,'sets')==0
	rawflag = 1;
    else
	rawflag = 0;
    end
else
    error('Failed to find %s',Fdata);
end

try
    	Fchannels = S.Fchannels;
catch
    	Fchannels = spm_select(1, '\.mat$', 'Select MEG channel template file', {}, fullfile(spm('dir'), 'EEGtemplates'));
end
if isempty(Fchannels) | ~exist(Fchannels)
   error(sprintf('Specified MEG template file %s does not exist',Fchannels))
else
   D.channels.ctf = Fchannels;
end

[Finter,Fgraph,CmdLine] = spm('FnUIsetup','MEG data conversion ',0);

% compatibility with some preprocessing functions
D.channels.heog = 0;
D.channels.veog = 0;
D.channels.reference = 0;
D.channels.ref_name = 'none';

%%%%%%%%%%%%%% Read channel information

[Channel,B] = spm_eeg_rdata_FIF_channels(Fdata);
D.Nchannels = length(Channel);

% Reorder channels by channel type (magnotometer vs gradiometer) 
% so that subsequent display in SPM is easier to select/deselect!

senstype = strvcat(Channel(:).SensType);
mags = strmatch('Magnetometer',senstype);
grds = strmatch('Gradiometer',senstype);
reord = [mags; grds];
Channel = Channel(reord);

% Detect Bad Channels (added later; only time megmodel is needed!)
try 
     emptyroom = S.emptyroom;
catch
     emptyroom = 0;
end

if ~emptyroom
     megmodel('head',Fdata);
     [bad,names] = badchans;
     bad = bad(reord);
     D.channels.Bad = find(bad);
else
     D.channels.Bad = [];
end

D.channels.eeg = 1:D.Nchannels;
D.channels.name = cat(1,Channel(:).Name);

Weights = cat(2,Channel(:).Weight)';
D.channels.Weight = reshape(Weights,2,length(Weights)/2)';
% spm_eeg_rdata_FIF_channels returns (gradiometer) Weights as 1/baseline, where baseline
% is in mm (eg 16.8), but spm_bst_headmodeler assumes SI units, ie m, so rescale to m:
D.channels.Weight(103:306,:) = D.channels.Weight(103:306,:)*1000;    

% Any scaling, eg to try to match mags and grds....
try
     sclchan = S.sclchan;
catch
     sclchan = 0;
end

if sclchan
    try
	sf = S.sclfact(:);
        if length(sf) ~= D.Nchannels
	    error(sprintf('Only %d scale factors supplied; %d expected',length(sf),D.Nchannels))
	else
	    disp('Using scale factors provided (assuming ordered by mags then grds...)')
	    D.channels.scaled = sf;
	end
     catch
	disp('Weighting gradiometers by SD of typical empty-room noise measurements (0.044)')
	disp('BEWARE: THE FORWARD MODEL IS NOT SCALED IN SAME WAY, SO YOU ARE MAKING A BIG ASSUMPTION IF YOU GO ON TO INVERT MAGS AND GRADS IN A SINGLE FILE (RATHER THAN USING SPM_EEG_INVERT_FUSE)')
        D.channels.scaled = ones(D.Nchannels,1);
	D.channels.scaled(103:306) = 0.044;
%     	D.channels.scaled = D.channels.Weight(:,1);
%	D.channels.Weight = D.channels.Weight./repmat(D.channels.scaled,1,size(D.channels.Weight,2));
     end
     disp('Will rescale channel data by...')
     disp(D.channels.scaled')
else
     D.channels.scaled = ones(D.Nchannels,1);
end

% Sensor positions and orientations not normally in D structure,
% but perhaps they should be!!! 
% (NB: These are properties of sensor array, ie helmet, so do not 
% depend on the subject, but their coordinates are in DEVICE space, 
% and need to be rotated to HEAD space, which does depend on subject)

if ~emptyroom
      Tmat = loadtrans(Fdata,'DEVICE','HEAD');
else
      Tmat = eye(4);
end
D.channels.Tmat = Tmat;

% Extract from Brainstorm 6xN format (using Brainstorm fields)....
CoilLoc = cat(2,Channel(:).Loc)';
CoilOrt = cat(2,Channel(:).Orient)';
GradOrt = cat(2,Channel(:).GradOrient)';

% Transform first coil info into head coordinates, and convert to mm
Loc = CoilLoc(:,1:3);
Ort = CoilOrt(:,1:3);
Loc = Tmat * [Loc ones(size(Loc,1),1)]';
Ort = Tmat * [Ort zeros(size(Ort,1),1)]';
D.channels.Loc = 1000*Loc(1:3,:);
D.channels.Orient = Ort(1:3,:);

% Use first coil for location and orientation of sensor itself (this is correct for mags; slight displacement for planar grads, but this only for display purposes; full coil location and orientation is retained in Loc and Orient fields)
D.channels.pos3D = D.channels.Loc(1:3,:)';
D.channels.ort3D = Tmat * [GradOrt(:,1:3) zeros(size(GradOrt,1),1)]';
D.channels.ort3D = D.channels.ort3D(1:3,:)';

% Transform second coil info into head coordinates (second coil in mags is NaN)
Loc = CoilLoc(:,4:6);
Ort = CoilOrt(:,4:6);
Loc = Tmat * [Loc ones(size(Loc,1),1)]';
Ort = Tmat * [Ort zeros(size(Ort,1),1)]';
D.channels.Loc = [D.channels.Loc; 1000*Loc(1:3,:)];
D.channels.Orient = [D.channels.Orient; Ort(1:3,:)];

% Could add option to get sensor locations from template, if not found in file
%D.channels.pos3D = Tmat * [Csetup.SensLoc ones(size(Csetup.SensLoc,1),1)]';
%D.channels.ort3D = Tmat * [Csetup.SensOr zeros(size(Csetup.SensOr,1),1)]';

% Read in Head Position Points (HPI) from Isotrak in HEAD space
% (If MaxMove has been used, these may have been deleted, so allow them to 
% be loaded seperately from a reference fif or mat file provided in S.HPIfile. 
% djm 27/6/07)
% jt 19/05/09 added friendly loop to test Fdata for HPI, prompt if none.

if ~emptyroom
 co=[];
 try
     HPIfile = S.HPIfile;
		 testHPI = 0
 catch
     HPIfile = Fdata;
		 testHPI = 1;
 end
 
 if testHPI  % Test whether Fdata FIF contains HPI points
   try
      [co,ki,nu] = hpipoints(HPIfile);
	 end
	 if isempty(co)
	   warning('FIF file does not contain HPI points!')
     HPIfile = spm_select(1,'any','Select FIF or MAT file w/ HPI points');
     [co,ki,nu] = hpipoints(HPIfile);
	 end
 else        % Get HPI from specified FIF/mat
  try 
    [pth name ext]=fileparts(HPIfile);
    if ext=='.fif'
        [co,ki,nu] = hpipoints(HPIfile);
    else
        load(HPIfile,'co','ki','nu');
    end
   end
 end 
 if isempty(co) | ~exist('ki') | ~exist('nu') | length(co)<7
    error('Failed to load HPI points!')
 end

 D.channels.fid_eeg = 1000*co(:,find(ki==1))';
 [dummy,nasion] = max(D.channels.fid_eeg(:,2));
 [dummy,LE] = min(D.channels.fid_eeg(:,1));
 [dummy,RE] = max(D.channels.fid_eeg(:,1));
 D.channels.fid_eeg = D.channels.fid_eeg([nasion LE RE],:);
 disp('Reordering cardinal points to...'); disp(D.channels.fid_eeg);
 D.channels.fid_coils = 1000*co(:,find(ki==2))';
 D.channels.headshape = 1000*co(:,find(ki==4))';

 try 
	pflag = S.pflag
 catch
	pflag = 1;
 end
 if pflag
  xyz = D.channels.pos3D;
  ori = D.channels.ort3D;
  fid = D.channels.fid_eeg;
  hsp = D.channels.headshape;
  nam = strvcat(D.channels.name{1:D.Nchannels});
  h    = spm_figure('GetWin','Graphics');
  clf(h); figure(h), hold on
  lstr = xyz(:,1:3)';
  lend = lstr+ori(:,1:3)'*10;
  plot3(xyz(:,1),xyz(:,2),xyz(:,3),'o');
%  t=text(xyz(1:102,1),xyz(1:102,2),xyz(1:102,3),nam(1:102,4:7));
  t=text(xyz(:,1),xyz(:,2),xyz(:,3),nam(:,4:7));
  set(t,'FontSize',6);
  line([lstr(1,:); lend(1,:)],[lstr(2,:); lend(2,:)],[lstr(3,:); lend(3,:)]);
  plot3(fid(:,1),fid(:,2),fid(:,3),'g+')
  plot3(hsp(:,1),hsp(:,2),hsp(:,3),'r.')
  title('Planar Grad locations slightly displaced for visualisation')
 end
 rotate3d on
end

%%%%%%%%%%%%%% Find channels in template for display

Csetup = load(D.channels.ctf);

for i = 1:D.Nchannels
  index = [];
  for j = 1:Csetup.Nchannels
    if ~isempty(find(strcmpi(D.channels.name{i}, Csetup.Cnames{j})))
      index = [index j];
    end
  end
  if isempty(index)
    warning(sprintf('No channel named %s found in channel template file.', D.channels.name{i}));
  else
    % take only the first found channel descriptor
    D.channels.order(i) = index(1);
  end
end

%%%%%%%%%%%%%% Any EOG?

B.eegfilt = find(B.chtypes==2 | B.chtypes==202 | B.chtypes==402);
Neog = 0; veog = -1; heog = -1; ecgc = -1; B.veogfilt=[]; B.heogfilt=[]; B.ecgfilt=[];


if ~isempty(B.eegfilt) 

  eegchan = strvcat(B.chans{B.eegfilt});
  eegchnums = str2num(eegchan(:,4:end));
  disp(sprintf('Found %d EEG/EOG channels',length(eegchnums)))

  try
     veog = S.veogchan(:);
  end
  while ~ismember(veog,[0; eegchnums])
    veog = spm_input('VEOG: chan nos [1-2; 0 for none]', '+1', 'r', [62], Inf, [0; eegchnums]);
  end
  if length(veog)>2
    error('Can only handle 1 or 2 VEOG channels')
  elseif length(veog)>1            % two unipolar channels?
    try
      subveog = S.veog_unipolar;
    catch
      subveog = spm_input('Subtract two VEOG channels (ie unipolar)?','+1','b',{'Yes|No'},[1 0],1);
    end
  else
      subveog = 0;
  end
  if veog(1)>0
   Neog = Neog+1;
   D.channels.veog = D.Nchannels+1;
   D.channels.name{D.channels.veog} = 'VEOG';
   [dummy,B.veognum] = intersect(eegchnums,veog); 
   B.veogfilt = B.eegfilt(B.veognum);
   
   index = [];
   for j = 1:Csetup.Nchannels
    if ~isempty(find(strcmpi('VEOG', Csetup.Cnames{j})))
      index = [index j];
    end
   end
   if isempty(index)
    warning(sprintf('No VEOG channel found in channel template file'));
   else
    % take only the first found channel descriptor
    D.channels.order(D.channels.veog) = index(1);
   end
  end

  try
     heog = S.heogchan(:);
  end
  while ~ismember(heog,[0; eegchnums])
    heog = spm_input('HEOG: chan nos [1-2; 0 for none]', '+1', 'r', [61], Inf, [0; eegchnums]);
  end
  if length(heog)>2
    error('Can only handle 1 or 2 HEOG channels')
  elseif length(heog)>1            % two unipolar channels?
    try
      subheog = S.heog_unipolar;
    catch
      subheog = spm_input('Subtract two HEOG channels (ie unipolar)?','+1','b',{'Yes|No'},[1 0],1);
    end    
  else
      subheog = 0;
  end
  if heog(1)>0
   Neog = Neog+1;
   D.channels.heog = D.Nchannels+2;
   D.channels.name{D.channels.heog} = 'HEOG';
   [dummy,B.heognum] = intersect(eegchnums,heog); 
   B.heogfilt = B.eegfilt(B.heognum);
   index = [];
   for j = 1:Csetup.Nchannels
    if ~isempty(find(strcmpi('HEOG', Csetup.Cnames{j})))
      index = [index j];
    end
   end
   if isempty(index)
    warning(sprintf('No HEOG channel found in channel template file'));
   else
    % take only the first found channel descriptor
    D.channels.order(D.channels.heog) = index(1);
   end
  end

  try
     ecgc = S.ecgchan(:);
  end
  while ~ismember(ecgc,[0; eegchnums])
    ecgc = spm_input('ECG: which EEG channel number(s) [1-2; 0 for none]', '+1', 'r', [63], Inf, [0; eegchnums]);
  end
  if length(ecgc)>2
    error('Can only handle 1 or 2 ECG channels')
  elseif length(ecgc)>1            % two unipolar channels?
    try
      subecg = S.ecg_unipolar;
    catch
      subecg = spm_input('Subtract two ECG channels (ie unipolar)?','+1','b',{'Yes|No'},[1 0],1);
    end
  else
      subecg = 0;
  end
  if ecgc(1)>0
   Neog = Neog+1;
   D.channels.ecg = D.Nchannels+3;
   D.channels.name{D.channels.ecg} = 'ECG';
   [dummy,B.ecgnum] = intersect(eegchnums,ecgc); 
   B.ecgfilt = B.eegfilt(B.ecgnum);
   
   index = [];
   for j = 1:Csetup.Nchannels
    if ~isempty(find(strcmpi('ECG', Csetup.Cnames{j})))
      index = [index j];
    end
   end
   if isempty(index)
    warning(sprintf('No ECG channel found in channel template file'));
   else
    % take only the first found channel descriptor
    D.channels.order(D.channels.ecg) = index(1);
   end
  end

  D.Nchannels = D.Nchannels + Neog;
  D.channels.scaled = [D.channels.scaled; ones(Neog,1)];
end

try 
        Dtype = S.Dtype;
catch
	Dtype = 'float32';
%	Dtype = 'int16';
end

disp(sprintf('\nReading and writing MEG channels (writing as %s)...\n',Dtype))

if rawflag == 0
%-----------------------------------
% IMPORT AVERAGE DATA:

 [B.totconds,B.comments] = loadfif(Fdata,'sets');
 disp(B.comments)

 try
	conds = S.conds;
	if conds == 0             % Flag to extract all automatically
	  conds = [1:B.totconds];
	end
 catch
	conds = spm_input(sprintf('Which conditions? (1-%d)',B.totconds),'+1','r',[1:B.totconds]);
 end

 D.Nevents = length(conds);
 D.events.types = [1:D.Nevents];
 D.events.Ntypes = length(D.events.types);
 D.events.code = [1:D.Nevents];
 D.events.reject = zeros(1, D.events.Ntypes);
 D.events.repl = ones(1, D.events.Ntypes);	% (lost original number of trials?)

%%%%%%%%%%%%%% Read data

 for c = conds			% Currently assumes B.t0 same for all conds
  [B.data{c},B.sfreq,B.t0] = loadfif(Fdata,c-1,'any');
  elen(c) = size(B.data{c},2);
 end
 D.Radc = B.sfreq;
 elen(elen==0)=[]; % djm 27/06/07
 disp('epoch lengths (in samples)...'),disp(elen)

%%%%%%%%%%%%%% Define epoch (S.twin, if specified, is eg [-100 300])

 B.t0 = round(B.t0*1000);	% convert to ms  % added round. djm 27/6/07. 
 swin = [1 min(elen)];
 try	
	twin = S.twin;
	if twin ==0           % Flag to extract all automatically
           twin = round((swin-1)*1000/D.Radc + B.t0);
	end
 catch
        twin = round((swin-1)*1000/D.Radc + B.t0);
	if any(diff(elen))
	  twin = spm_input('Sample window? (ms)','+1','r',twin,2,twin)';
	end
 end
	
 swin = round((twin-B.t0)*D.Radc/1000)+1;
 if length(twin)~=2 | twin(1) < B.t0 |  twin(1) > -1 | swin(2) > min(elen)
	error('twin outside range for all conditions')
 end

 D.Nsamples = swin(2) - swin(1) + 1;
 D.events.start = -round(twin(1)*D.Radc/1000);
 D.events.stop = swin(2) - swin(1) - D.events.start;

%%%%%%%%%%%%%% Reformat data
% !!Could do baseline correction here, particularly if baseline period changed
 
 d = [];
 for c = 1:length(conds)
   d(:,:,c) = B.data{conds(c)}(B.chanfilt,swin(1):swin(2));
 end

 d = d(reord,:,:);

 d = d*10^15;		% convert to fT units 

%%%%%%%%%%%%%% Prepare output file

 try
    % option to provide different output file in S.Pout - djm 27/6/07
    [f1, f2, f3] = fileparts(S.Pout);
    D.fname = [f2 '.mat'];      
    D.fnamedat = [f2 '.dat']; 
    if ~isempty(f1); D.path = f1; end;
 catch
    [dummy,stem,ext] = fileparts(Fdata);
    D.fname = strcat('me_',stem,'.mat');
    D.fnamedat = strcat('me_',stem,'.dat');
 end

%%%%%%%%%%%%%% Add EOG data (if any) 
 dd=[];
 if veog(1)>0
   for c = 1:length(conds)
    deog = B.data{conds(c)}(B.veogfilt,swin(1):swin(2))*10^6;  % postmultiplier is to convert to uV

    if subveog
      deog = [1 -1]*deog;	% Create bipolar data
    else
      deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
    end
    dd(:,c) = deog';
   end
   d(end+1,:,:) = dd;
 end

 dd=[];
 if heog(1)>0
   for c = 1:length(conds)
    deog = B.data{conds(c)}(B.heogfilt,swin(1):swin(2))*10^6;  % postmultiplier is to convert to uV
    if subheog
      deog = [1 -1]*deog;	% Create biabspolar data
    else
      deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
    end
    dd(:,c) = deog';
   end
   d(end+1,:,:) = dd;
 end
 
 dd=[];
 if ecgc(1)>0
   for c = 1:length(conds)
    deog = B.data{conds(c)}(B.ecgfilt,swin(1):swin(2))*10^6;  % postmultiplier is to convert to uV

    if subecg
      deog = [1 -1]*deog;	% Create bipolar data
    else
      deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
    end
    dd(:,c) = deog';
   end
   d(end+1,:,:) = dd;
 end

 d = d.*repmat(D.channels.scaled,[1 size(d,2) size(d,3)]);

%%%%%%%%%%%%%% Write average data to *.dat file

 spm('Pointer', 'Watch'); drawnow;

 D.scale = ones(D.Nchannels, 1, D.Nevents);
 D.datatype  = Dtype;

 fpd = fopen(fullfile(D.path, D.fnamedat), 'w');

 spm_progress_bar('Init', 100, 'Events written'); drawnow;
 if length(D.Nevents) > 100, Ibar = floor(linspace(1, D.Nevents,100));
 else, Ibar = [1:D.Nevents]; end

 for e = 1:D.Nevents
      for s = 1:D.Nsamples	
	fwrite(fpd, d(:,s,e), Dtype);
      end

      barh = find(Ibar==e);
      if ~isempty(barh)
           spm_progress_bar('Set', barh);
           drawnow;
      end
 end

 fclose(fpd);


else
%-----------------------------------
% IMPORT RAW DATA:

%%%%%%%%%%%%%% Find trigger channel
   D.Nevents = 1;
   try 
	trig_chan = S.trig_chan;
   catch
	trig_chan = spm_input('Trigger channel name?(0=none)','+1','s','STI101');
   end
   disp(sprintf('Trigger channel %s',trig_chan))

   if ~strcmp(trig_chan,'0')
    k = [];
    while isempty(k)
     c = 1;
     while isempty(k) & c <= size(B.chans,1)
	k = findstr(B.chans{c},trig_chan);
	c = c+1;
     end
     if c > size(B.chans,1)
	disp(B.chans)
	disp('Channel not found...')
	trig_chan = spm_input('Trigger channel name?(0=none)?','+1','s','STI101');
     end
    end

    try 
	trig_type = S.trig_type;
    catch
	trig_type = 'positive_change'; % Could prompt user instead
    end
    disp(sprintf('Trigger type %s',trig_type))

%%%%%%%%%%%%%% Read events

   spm('Pointer', 'Watch'); drawnow;
   [B.data, D.Radc] = rawchannels(Fdata,trig_chan,'noskips');
   varargout{1} = B.data;
   D.Nsamples = size(B.data,2);

   if ~isempty(regexp(trig_type,'recode_buttons','ONCE'))
       % Button press events can summate and send variable values to ST101, so get
       % these from dedicated channels and code 1st sample arbitrarily as
       % 1000 + channel number. Then remove the button press from subsequent
       % samples on trigger channel so it doesn't interfere with other triggers
       fprintf('\nChecking for button presses on dedicated channels')
       buttonmask=zeros(size(B.data));
       for c=9:16
           chname=sprintf('STI%03g',c);
           if sum(strcmp(chname,B.chans))==1
               [presses] = rawchannels(Fdata,chname);
               buttonvalue=mode(B.data(presses>0));
               fprintf('..STI%03g(%g)',c,buttonvalue)
               % remove button press events from main trigger channel:
               B.data(presses>0)=B.data(presses>0)-buttonvalue;
               % add new code for first two buttonpress samples to mask:
               onsets=find(presses.*[0 diff(presses)]);
               buttonmask([onsets, onsets+1])=1000+c;
           end
       end
       clear presses events values
       % overwrite recoded button press onsets to trigger channel:
       B.data(buttonmask>0)=buttonmask(buttonmask>0);
   end

   if strcmp(trig_type,'positive_change')
% If assume trigger onset always positive deflection (will fail if risetime > 1 sample)
     D.events.time = find(diff(B.data)>0)+1;
   elseif strcmp(trig_type,'any_change')
% If assume trigger onset always positive deflection (will fail if risetime > 1 sample)
     D.events.time = find(abs(diff(B.data))>0)+1;
   elseif strcmp(trig_type,'every_other_change')
% Suitable if, eg 1) first sample is 0; 2) 0's between triggers, 3) triggers at least two samples in duration
     trig = find(abs(diff(B.data))>0)+1;
     D.events.time = trig(1:2:end);
   elseif strcmp(trig_type,'positive_from_zero')
% If assume trigger onset always positive deflection from zero (will fail if risetime > 1 sample)
     trig = find(diff(B.data)>0)+1;
     D.events.time = trig(find(B.data(trig-1)==0));
   elseif strcmp(trig_type,'positive_from_zero')
% If assume trigger onset always positive deflection from zero (will fail if risetime > 1 sample)
     trig = find(diff(B.data)>0)+1;
     D.events.time = trig(find(B.data(trig-1)==0));
   elseif strcmp(trig_type,'any_change_from_zero')
% If assume trigger onset always a deflection from zero (will fail if risetime > 1 sample)
     trig = find(diff(B.data)~=0)+1;
     D.events.time = trig(find(B.data(trig-1)==0));
   elseif ~isempty(regexp(trig_type,'new_stable_change','ONCE'))
% Remove transients that last for one sample ...
     for ml=1:3
        offending=find(diff(B.data).*circshift(diff(B.data),[1 1]));
        B.data(offending)=B.data(offending+1);
     end
     % ...then define an event for any change on trigger channel that doesn't go to 0 or to previous value
     events=find(diff(B.data)~=0)+1;
     values=B.data(events);
     events(values==0|(values-[NaN NaN values(1:end-2)]==0))=[];
     D.events.time=events;
   elseif isnumeric(trig_type)
% Only detect a specific trigger code (one number)
     trig = find(B.data==trig_type);
     D.events.time = trig([1 find(diff(trig)>1)+1]);
   else
     error('Unknown trigger type?!')
   end
  
   disp(sprintf('%d triggers found...',length(D.events.time)))
   if length(D.events.time) > 0
    D.events.code = B.data(D.events.time);
    D.events.types = unique(D.events.code);
    disp(sprintf('Trigger codes = %s',mat2str(D.events.types)))
    D.events.Ntypes = length(D.events.types);
    for eve = 1:D.events.Ntypes
      numeve(eve) = length(find(D.events.code==D.events.types(eve)));
    end
    disp(sprintf('No. each code = %s',mat2str(numeve)))
   else % No triggers found, so treat as one event-type
    D.events.time = 1; D.events.code = 1; 
   end
  
  else 
   spm('Pointer', 'Watch'); drawnow;
   [B.data, D.Radc] = rawchannels(Fdata,B.chans{1});
   D.Nsamples = size(B.data,2);
   D.events.time = 1;
   D.events.code = 1;
  end

 spm('Pointer', 'Arrow'); drawnow;

%%%%%%%%%%%%%% Select sample
 try	
	twin = S.twin;
 catch
	swin = [1 D.Nsamples];
	twin = round(1000*(swin-1)/D.Radc);
	twin = spm_input('Sample window? (ms)','+1','r',twin,2,twin)';
 end

% If want to prespecify all, without precise knowledge of length
 if twin(2) == Inf	
	swin(1) = round(twin(1)*D.Radc/1000)+1;
	swin(2) = D.Nsamples;
 else
	swin = round(twin*D.Radc/1000)+1;
 end
 
 if length(twin)~=2 | twin(1) < 0 | swin(2) > D.Nsamples
	error('sample outside range')
 end

% Keep all events & times or only those within time-window & resynch time? 
%   Only if time-window specified (jt 09/07/09)
if ~(swin(1)==1 & swin(2)==D.Nsamples)
	try   twinevents = S.twinevents;
	catch twinevents = spm_input('Keep events in twin or all?','+1','b',{'twin|all'},[1 0],1);
	end
	
  if twinevents
    try   doresynch = S.resynch;
  	catch doresynch = spm_input('Resynch time to twin?','+1','b',{'yes|no'},[1 0],1);
  	end
	  keepev = find(D.events.time>=swin(1) & D.events.time<=swin(2));
 	  D.events.code   = D.events.code(keepev);
		D.events.time   = D.events.time(keepev);
		if doresynch
		  D.events.time = D.events.time-swin(1)+1;
		end
 	  D.events.types  = unique(D.events.code);
 	  D.events.Ntypes = length(D.events.types);
	end

end

%%%%%%%%%%%%%% Prepare output file

 try
    % option to provide different output file in S.Pout - djm 27/6/07
    [f1, f2, f3] = fileparts(S.Pout);
    D.fname = [f2 '.mat'];      
    D.fnamedat = [f2 '.dat'];      
    if ~isempty(f1); D.path = f1; end;
 catch
    [dummy,stem,ext] = fileparts(Fdata);
    D.fname = strcat(stem,'.mat');
    D.fnamedat = strcat(stem,'.dat');
 end

 fpd = fopen(fullfile(D.path, D.fnamedat), 'w');

%%%%%%%%%%%%%% Read and write raw data (in blocks)

 spm('Pointer', 'Watch'); drawnow;

 spm_progress_bar('Init', 100, 'Samples written'); drawnow;
 Ibar = floor(linspace(1, swin(2), 100));
 status='ok'; lfi=0;
 
 dl=0;
 rawdata('any',Fdata);
 [rd,status] = rawdata('next');
 while ~(strcmp(status,'eof') | strcmp(status,'error'))
   if strcmp(status,'ok')
    di = [1:size(rd,2)]+dl;
    dl = di(end);
    di = find(di>=swin(1) & di<=swin(2));
   
    if ~isempty(di)	
	d = rd(B.chanfilt,di);
        d = d(reord,:)*10^15;

%%%%%%%%%%%%%% Add EOG data (if any) 

 	if veog(1)>0
    	  deog = rd(B.veogfilt,di)*10^6;
    	  if subveog
           deog = [1 -1]*deog;	% Create bipolar data
          else
           deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
          end
          d(end+1,:) = deog;
        end
 	if heog(1)>0
    	  deog = rd(B.heogfilt,di)*10^6;
    	  if subheog 
           deog = [1 -1]*deog;	% Create bipolar data
          else
           deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
          end
          d(end+1,:) = deog;
        end
 	if ecgc(1)>0
    	  deog = rd(B.ecgfilt,di)*10^6;
    	  if subecg
           deog = [1 -1]*deog;	% Create bipolar data
          else
           deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
          end
          d(end+1,:) = deog;
        end

        d = d.*repmat(D.channels.scaled,1,size(d,2));

        for s = 1:length(di)
	    fwrite(fpd, d(:,s), Dtype);
        end
    end

    fi = find((Ibar-dl) > 0);
    if isempty(fi)
	  status = 0;
    elseif fi(1)>lfi
	  lfi=fi(1);
          spm_progress_bar('Set', lfi);
          drawnow;
    end
   else
    disp(sprintf('Encountered a skip at sample %d',dl))
   end

   [rd,status] = rawdata('next');
 end
 rawdata('close');

 fclose(fpd);

 D.Nsamples = swin(2) - swin(1) + 1;
 disp(sprintf('%d samples read (%f seconds)',D.Nsamples,D.Nsamples/D.Radc))

 D.scale = ones(D.Nchannels, 1);
 D.datatype  = Dtype;

end

%%%%%%%%%%%%%% save *.mat file (for both ave and raw)

D.units = 'fT';
D.modality = 'MEG';

spm_progress_bar('Clear');

if str2num(version('-release'))>=14 
    save(fullfile(D.path, D.fname), '-V6', 'D');
else
    save(fullfile(D.path, D.fname), 'D');
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Now check whether extra EEG channels need to be saved in new file

B.noneogfilt = setxor(B.eegfilt,[B.veogfilt; B.heogfilt; B.ecgfilt]);

if ~isempty(B.noneogfilt)

  disp(sprintf('\nReading and writing EEG channels...\n'))

  D2 = D;
  D2.Nchannels = size(eegchan,1);
  D2.channels = rmfield(D.channels,{'name','Weight','scaled','Loc','Orient','pos3D','ort3D'});
  D2.units = 'uV';
  D2.modality = 'EEG';

% Rename channels 1-128 (in case want to use CNT128 EEGtemplate?)
%  D2.channels.name = mat2cell(eegchan(:,4:6),ones(1,D2.Nchannels),3); 
%  D2.channels.name = cell(D2.Nchannels,1);

  cc=0; veogflag=0; heogflag=0; ecgflag=0;
  for c=1:D2.Nchannels
    if ismember(eegchnums(c),veog)
      veogflag=1;
    elseif ismember(eegchnums(c),heog)
      heogflag=1;
    elseif ismember(eegchnums(c),ecgc)
      ecgflag=1;
    else
      cc = cc+1;
      D2.channels.name{cc,1} = num2str(str2num(eegchan(c,4:6)));
    end
  end
  if veogflag, D2.channels.name{end+1} = 'VEOG'; end
  if heogflag, D2.channels.name{end+1} = 'HEOG'; end
  if ecgflag, D2.channels.name{end+1} = 'ECG'; end
  D2.Nchannels = length(D2.channels.name); % size(D2.channels.name,1);
  D2.channels.eeg = [1:(D2.Nchannels-veogflag-heogflag-ecgflag)];
  if veogflag;  D2.channels.veog = D2.Nchannels-heogflag-ecgflag;  end
  if heogflag;  D2.channels.heog = D2.Nchannels-ecgflag;  end  
  if ecgflag;  D2.channels.ecg = D2.Nchannels;  end  	
  D2.channels.order = [];
  D2.channels.scaled = ones(D2.Nchannels,1);  % No scaling option for EEG yet
  
%%%%%%%%%%%%%%%%%%%% Display electrodes (all digitised)

  xyz = 1000*co(:,find(ki==3))';

  if pflag
 	   fid = D2.channels.fid_eeg;
 	   hsp = D2.channels.headshape;
 	   nam = strvcat(D2.channels.name{D2.channels.eeg});
 	   h    = spm_figure('GetWin','Graphics');
 	   clf(h); figure(h), hold on
	   plot3(xyz(:,1),xyz(:,2),xyz(:,3),'o');
 	   t=text(xyz(:,1),xyz(:,2),xyz(:,3), num2str([1:size(xyz,1)]'));
   	   set(t,'FontSize',6);
    	   plot3(fid(:,1),fid(:,2),fid(:,3),'g+')
           plot3(hsp(:,1),hsp(:,2),hsp(:,3),'r.')
    	   title('Electrode locations in order digitized')
           rotate3d on
  end

% Assumes first electrode digitised is reference, so remove
  try
    refchan = S.refchan;
  catch
    refchan = 1;
  end
  xyz(refchan,:) = [];
  
  
% In old digitisation method of MEG lab, EOG and ECG were marked as EEG
% in acquisition setup file, which meant that 61-64 had to be digitised
% too. In new method, 61-64 not marked as EEG (marked as EOG,ECG or off),
% so digitisation can proceed sequentially beyond 70

  if D2.Nchannels > 60
   try
    dig_method = S.dig_method;
   catch
    dig_method = spm_input('Old (0) or new (1) digitisation', '+1', 'i', [1], Inf, [0; 1]);
   end
   
   if ~dig_method
    rmchans=[];
    if veogflag; rmchans = veog;  end
    if heogflag; rmchans = [rmchans; heog];  end
    if ecgflag; rmchans = [rmchans; ecgc];  end		
    xyz(rmchans,:) = []; 
   end
  end
  
  disp(sprintf(['!!!Removed 1st (reference) and any eog/ecg electrodes digitised (leaving %d)!!!'],size(xyz,1)))
  D2.channels.Loc = xyz';
  
  try
     Fchannels = S.Fchannels_eeg;
  catch
     Fchannels = spm_select([0 1], '\.mat$', ['Select EEG channel template file (or none to create your own)'], {}, fullfile(spm('dir'), 'EEGtemplates'));
  end
  
%%%%%%%%%%%%%%%%%%%% Use or make a template for EEG display
  if ~isempty(Fchannels) & exist(Fchannels)
      disp(sprintf('Using existing EEG template %s',Fchannels)); 
      D2.channels.ctf = Fchannels;
  else
      if isempty(Fchannels)
	Fchannels = 'meg_eeg_montage';
      elseif ~exist(Fchannels)
        warning(sprintf('Specified EEG template file %s does not exist',Fchannels))
      end
      
      disp(sprintf('Creating new EEG template %s...',Fchannels));
      
% Need to know which channel name (1-128) corresponds to Cz!!! (may not always be max in Z)
        warning('!!!Assuming Cz is electrode with maximal z....!!!')
  	[zmax,cz] = max(xyz(:,3));
  	coord = xyz - repmat(xyz(cz,:)+[0 0 250],size(xyz,1),1);
  	d = sqrt(sum(coord'.^2));
% projection to x-y-plane by scaling by distance
  	coord = coord.*repmat(d.^2', 1,3);
  	x = coord(:,1);  	y = coord(:,2);
  	mx = max(abs(x));  	my = max(abs(y));
% make coordinates lie between 0.05 and 0.95 for x and y
  	x = x./(2*mx)*0.9 + 0.5;  y = y./(2*my)*0.9 + 0.5;

  	Cpos = [x y]';
  	Nchannels = size(Cpos,2);
%  	Cnames = strtrim(mat2cell(num2str([1:Nchannels]'),ones(1,Nchannels),3));
 	tmp = strvcat(D2.channels.name);
        Cnames = strtrim(mat2cell(tmp(D2.channels.eeg,1:3),ones(1,Nchannels),3));
 	Cpos = [Cpos [0.1; 0.95]];  Cnames{end+1} = 'VEOG';
  	Cpos = [Cpos [0.9; 0.95]];  Cnames{end+1} = 'HEOG';
 	Cpos = [Cpos [0.8; 0.95]];  Cnames{end+1} = 'ECG';
  	Nchannels = size(Cpos,2);

  	if pflag 
		figure,plot(Cpos(1,:)',Cpos(2,:)','o')
  		t=text(Cpos(1,:)',Cpos(2,:)',Cnames);
	end

% display ratio of x- and y-axis (used by SPM's display of M/EEG data)
  	Rxy = 1.5;

%%%% Need write permission for your SPM installation!
        D2.channels.ctf = Fchannels;
  	save(D2.channels.ctf,'Cpos','Cnames','Rxy','Nchannels')
  end

%%%%%%%%%%%%%% Find channels in template for display

  Csetup = load(D2.channels.ctf);
  for i = 1:D2.Nchannels
   index = [];
   for j = 1:Csetup.Nchannels
    if ~isempty(find(strcmpi(D2.channels.name{i}, Csetup.Cnames{j})))
      index = [index j];
    end
   end
   if isempty(index)
    warning(sprintf('No channel named %s found in channel template file.', D2.channels.name{i}));
   else
    % take only the first found channel descriptor
    D2.channels.order(i) = index(1);
   end
  end
  
  D2.channels.Bad = [];

%%%%%%%%%%%%%%% Sort out reference 
  try 
	D2.channels.ref_name = S.eeg_ref;
	if strcmp(S.eeg_ref,'average')		% reserved term!
	 D2.channels.reference = D2.channels.order(D2.channels.eeg);
	else
       	 index = [];
   	 for j = 1:Csetup.Nchannels
    	  if ~isempty(find(strcmpi(S.eeg_ref, Csetup.Cnames{j})))
      		index = [index j];
    	  end
   	 end
  	 if isempty(index)
    	  warning(sprintf('No channel named %s (given as reference) in channel template file.', D2.channels.name{i}))
          D2.channels.reference = 0;
   	 else
    	  D2.channels.reference = index(1);
         end
        end
  catch
  	D2.channels.ref_name = 'WhateverYouChose';
  	D2.channels.reference = 0;
  end


  if rawflag == 0
%-----------------------------------
% IMPORT AVERAGE DATA:

%%%%%%%%%%%%%% Reformat data
% !!Could do baseline correction here, particularly if baseline period changed

 d=[];
 for c = 1:length(conds)
   d(:,:,c) = B.data{conds(c)}(B.noneogfilt,swin(1):swin(2));
 end

 d = d*10^6;	 	                     % convert to uV units 

 if strcmp(D2.channels.ref_name,'average')   % Re-ref to average if requested
   d = detrend(d,0);
 end

%%%%%%%%%%%%%% Prepare output file

 try
    % option to provide different output file in S.Pout - djm 27/6/07
    [f1, f2, f3] = fileparts(S.Pout);
    D2.fname = [f2 '-eeg.mat'];      
    D2.fnamedat = [f2 '-eeg.dat']; 
    if ~isempty(f1); D2.path = f1; end;
 catch
    [dummy,stem,ext] = fileparts(Fdata);
    D2.fname = strcat('me_',stem,'-eeg.mat');
    D2.fnamedat = strcat('me_',stem,'-eeg.dat');
 end

%%%%%%%%%%%%%% Add EOG data (if any)
 
 dd=[];
 if veog(1)>0
   for c = 1:length(conds)
    deog = B.data{conds(c)}(B.veogfilt,swin(1):swin(2))*10^6;  % postmultiplier is to convert to uV

    if subveog
     deog = [1 -1]*deog;	% Create bipolar data
    else
     deog = deog(1,:);          % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
    end
    dd(:,c) = deog';
   end
   d(end+1,:,:) = dd;
 end

 dd=[];
 if heog(1)>0
   for c = 1:length(conds)
    deog = B.data{conds(c)}(B.heogfilt,swin(1):swin(2))*10^6;  % postmultiplier is to convert to uV
    if subheog
     deog = [1 -1]*deog;	% Create bipolar data
    else
     deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
    end
    dd(:,c) = deog';
   end
   d(end+1,:,:) = dd;
 end
 
 dd=[];
 if ecgc(1)>0
   for c = 1:length(conds)
    deog = B.data{conds(c)}(B.ecgfilt,swin(1):swin(2))*10^6;  % postmultiplier is to convert to uV

    if subecg
     deog = [1 -1]*deog;	% Create bipolar data
    else
     deog = deog(1,:);          % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
    end
    dd(:,c) = deog';
   end
   d(end+1,:,:) = dd;
 end

%%%%%%%%%%%%%% Write average data to *.dat file

 spm('Pointer', 'Watch'); drawnow;

 D2.scale = ones(D2.Nchannels, 1, D2.Nevents);
 D2.datatype  = Dtype;

 fpd = fopen(fullfile(D2.path, D2.fnamedat), 'w');

 spm_progress_bar('Init', 100, 'Events written'); drawnow;
 if length(D2.Nevents) > 100, Ibar = floor(linspace(1, D2.Nevents,100));
 else, Ibar = [1:D2.Nevents]; end

 for e = 1:D2.Nevents
      for s = 1:D2.Nsamples	
          fwrite(fpd, d(:,s,e), Dtype);
      end

      barh = find(Ibar==e);
      if ~isempty(barh)
           spm_progress_bar('Set', barh);
           drawnow;
      end
 end

 fclose(fpd);


 else
%-----------------------------------
% IMPORT RAW DATA:

%%%%%%%%%%%%%% Prepare output file

 try
    % option to provide different output file in S.Pout - djm 27/6/07
    [f1, f2, f3] = fileparts(S.Pout);
    D2.fname = [f2 '-eeg.mat'];      
    D2.fnamedat = [f2 '-eeg.dat'];      
    if ~isempty(f1); D2.path = f1; end;
 catch
    [dummy,stem,ext] = fileparts(Fdata);
    D2.fname = strcat(stem,'-eeg.mat');
    D2.fnamedat = strcat(stem,'-eeg.dat');
 end

 fpd = fopen(fullfile(D2.path, D2.fnamedat), 'w');

%%%%%%%%%%%%%% Read and write raw data (in blocks)
% OK, OK, so not very efficient to read it all in again, but saves memory!

 spm('Pointer', 'Watch'); drawnow;

 spm_progress_bar('Init', 100, 'Samples written'); drawnow;
 Ibar = floor(linspace(1, swin(2), 100));
 status='ok'; lfi=0;
 
 dl=0;
 rawdata('any',Fdata);
 [rd,status] = rawdata('next');
 while ~(strcmp(status,'eof') | strcmp(status,'error'))
   if strcmp(status,'ok')
    di = [1:size(rd,2)]+dl;
    dl = di(end);
    di = find(di>=swin(1) & di<=swin(2));
   
    if ~isempty(di)	
	d = rd(B.noneogfilt,di)*10^6;

 	if strcmp(D2.channels.ref_name,'average')
	    d = detrend(d,0);   % Re-ref to average if requested
	end

%%%%%%%%%%%%%% Add EOG data (if any) 

 	if veog(1)>0
    	  deog = rd(B.veogfilt,di)*10^6;
    	  if subveog
           deog = [1 -1]*deog;	% Create bipolar data
          else
           deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
          end
          d(end+1,:) = deog;
        end
 	if heog(1)>0
    	  deog = rd(B.heogfilt,di)*10^6;
    	  if subheog
           deog = [1 -1]*deog;	% Create bipolar data
          else
           deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
          end
          d(end+1,:) = deog;
        end
	if ecgc(1)>0
    	  deog = rd(B.ecgfilt,di)*10^6;
    	  if subecg
           deog = [1 -1]*deog;	% Create bipolar data
          else
           deog = deog(1,:);         % or take first channel (eg if 124 EEG
                                % channels require digitising 61-64 for EOG)
          end
          d(end+1,:) = deog;
        end

        for s = 1:length(di)
	    fwrite(fpd, d(:,s), Dtype);
        end
    end

    fi = find((Ibar-dl) > 0);
    if isempty(fi)
	  status = 0;
    elseif fi(1)>lfi
	  lfi=fi(1);
          spm_progress_bar('Set', lfi);
          drawnow;
    end
   else
    disp(sprintf('Encountered a skip at sample %d',dl))
   end

   [rd,status] = rawdata('next');
 end
 rawdata('close');

 fclose(fpd);

 D2.Nsamples = swin(2) - swin(1) + 1;
 if D.Nsamples ~= D2.Nsamples
   error('Different number of MEG and EEG samples!!???')
 end
 disp(sprintf('%d samples read (%f seconds)',D2.Nsamples,D2.Nsamples/D2.Radc))

 D2.scale = ones(D2.Nchannels, 1);
 D2.datatype  = Dtype;

 end
  
Dtmp = D;
D=D2;
spm_progress_bar('Clear');

if str2num(version('-release'))>=14 
    save(fullfile(D.path, D.fname), '-V6', 'D');
else
    save(fullfile(D.path, D.fname), 'D');
end
D = Dtmp;

end


%%%%%%%%%%%%
spm('Pointer', 'Arrow');

rotate3d off

return
