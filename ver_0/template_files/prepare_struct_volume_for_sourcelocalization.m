% MAKE SURE TO RESPOSITION STRUCTURALS FIRST, and CHECK SEGMENTATIONS
% AFTER!

clear all

subname = '08_0480';

%% setup

fieldtripdefs % adds subfolders

spm2path = '/imaging/local/spm/spm2';
spm8path = '/imaging/local/spm/spm8';
spm5path = '/imaging/local/spm/spm5';

%addpath(spm2path);


% go to the data directory
structdir = fullfile('/imaging/jp01/experiments/speech_synchronization/structurals/', subname, 'structurals');
cd(structdir);

%don't use headshape, only 3 fiducials
addpath(spm5path);

%mri = spm_select(1, 'nii', sprintf('Select structural for %s', subname), [], structdir, '^s');
mri = spm_select('fplist', structdir, '^s.*\.nii$')
if strcmp(mri, '/')
    error('No mri found.\n');
else
    fprintf('Found MRI: %s\n', mri);
end

mri   = read_mri(mri);
cfg = '';
cfg.method = 'interactive';

rmpath(spm5path);
addpath(spm2path);
spm('fmri'); close all

mri2=volumerealign(cfg,mri); %left at top!



%mri UL: coronal, neck right;; UR: axial eyes left;; LL: sagitall, nose
%down
%headshape: x: nose, y: ear. z: top

%work in normalized space
cfg='';
cfg.template= fullfile(spm2path, 'templates/T1.mnc');
cfg.nonlinear   = 'no';
cfg.coordinates = 'ctf';  % ??
cfg.write = 'yes';
cfg.name = 'norm';
[normalise] = volumenormalise(cfg, mri2);
normtrans=normalise.cfg.final;



%% prepare singleshell model
cfg             = [];
cfg.coordinates = 'spm';
cfg.template    = fullfile(spm2path, 'templates/T1.mnc');
segment         = volumesegment(cfg, normalise);

cfg = [];
%cfg.headshape = ''; % this gave an error
vol = prepare_singleshell(cfg, segment);
vol = convert_units(vol, 'cm');
vol.bnd.pnt = warp_apply(inv(normtrans), vol.bnd.pnt*10, 'homogenous')/10;

% save the structural information to the structural directory
outname = fullfile(structdir,subname);
fprintf('Saving to %s...', outname);
save(outname, 'vol', 'normtrans', 'mri');
close all
fprintf('done.\n');

