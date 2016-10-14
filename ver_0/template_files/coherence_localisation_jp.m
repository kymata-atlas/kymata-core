function coherence_localisation_jp(subname, options)
%COHERENCE_LOCALISATION_JP(SUBNAME, [CFG])
%
% cfg.analysisname
% cfg.lambda (e.g. '1%')

if nargin < 2
    options = [];
end


if ~isfield(options, 'analysisname')
    error('Must specify cfg.analysisname');
end

if ~isfield(options, 'lambda') || ~ischar(options.lambda);
    error('Must specify cfg.lambda as a string');
end

analysisname = options.analysisname;

datadir = fullfile('/imaging/jp01/experiments/speech_synchronization/subj/', subname, 'analysis_sentences_nocleaning');
outputdir = fullfile(datadir, analysisname);
if ~isdir(outputdir)
    mkdir(outputdir);
end
structdir = fullfile('/imaging/jp01/experiments/speech_synchronization/structurals/', subname, 'structurals');

spm2path = '/imaging/local/spm/spm2';
spm8path = '/imaging/local/spm/spm8';
spm5path = '/imaging/local/spm/spm5';


fieldtripdefs % adds subfolders

% get info from structural processing
load(fullfile(structdir,subname) , 'vol', 'normtrans', 'mri');

fname = '/imaging/jp01/experiments/speech_synchronization/templategrid6mm';
load(fname, 'grid');

posmni=grid.pos;
pos             = warp_apply(inv(normtrans), grid.pos*10, 'homogenous')/10;
grid.pos        = pos;

conds = {'16ch' '04ch' '01ch' '04ch_rot'};


for c=1:length(conds)
    
    rmpath(spm2path);
    addpath(spm5path);
    spm('fmri'); close all
    
    thiscond = conds{c};

    fprintf('\n\nSource localization for condition %s...\n', thiscond);

    %% CTF coordinate system
    %use ctf coordinate system  %<-- is this right for neuromag data? -jp
    load(fullfile(datadir, sprintf('allblockdata_%s', thiscond)));
    grad=convert_units(allblockdata.grad,'cm');
    grad2=grad;
    grad2.pnt(:,1)=grad.pnt(:,2);
    grad2.pnt(:,2)=-grad.pnt(:,1);
    grad2.ori(:,1)=grad.ori(:,2);
    grad2.ori(:,2)=-grad.ori(:,1);


    %% different coordinate system. neuromag is x-left, y-front, z-top
    cfg=[];
    cfg.vol=vol;
    cfg.grid=grid;
    cfg.grad=grad2;
    headmodelplot(cfg);

    allblockdata.grad=grad2;

    %% source localization
    cfg             = [];
    cfg.grid        = grid;
    cfg.grad        = grad2;
    cfg.vol         = vol;
    cfg.channel     = {'MEG'};
    cfg.normalize   = 'column';
    cfg.feedback    ='none';
    grid            = prepare_leadfield(cfg,allblockdata);

    %% get csd
    data=allblockdata;

    %% freqanalysis
    cfg = [];
    cfg.output = 'powandcsd';
    cfg.method = 'mtmfft';
    cfg.taper = 'dpss';
    cfg.foilim = [4 8];
    cfg.tapsmofrq = 1;
    %cfg.keeptrials = 'yes';
    cfg.channel = {'MEG' 'ENV'};
    freq = freqanalysis(cfg, data);


    cfg           = [];
    cfg.method    = 'pcc';
    cfg.frequency = 6;
    cfg.vol       = vol;
    cfg.grad      = grad2;
    cfg.grid      = grid;
    cfg.feedback  = 'none';
    cfg.normalize = 'no';
    %cfg.keepfilter= 'yes';
    cfg.lambda    = options.lambda; %'1%';
    %cfg.channel   = {'*3'};%{'*1','*2'}
    cfg.refchan   = 'ENV';
    source        = sourceanalysis(cfg, freq);

    sd = sourcedescriptives([], source);

    %% get ready for source interpolation

    
    mri = read_mri(fullfile(spm5path, 'canonical/avg152T1.nii'));
    sd.pos = posmni;

    cfg=[];
    cfg.parameter = 'coh';
    cfg.downsample = 2;
    [interp] = sourceinterpolate(cfg,sd,mri);
    

    %% finish up

    cfg=[];
    cfg.method='ortho';
    cfg.anaparameter='anatomy';
    cfg.interactive='no';
    cfg.funparameter='coh';
    figure('color', 'w', 'name', sprintf('%s: %s', subname, thiscond));
            
    sourceplot(cfg,interp);

    % write the volume
    fprintf('\tWriting volume...');
    
    rmpath(spm5path);
    addpath(spm2path);
    spm('fmri'); close all
    
    cfg = [];
    cfg.parameter = 'coh';
    cfg.scaling = 'no';
    cfg.datatype = 'double';
    cfg.filename = fullfile(outputdir, sprintf('%s_coherence_%s', subname, thiscond));
    volumewrite(cfg, interp);
    fprintf('done.\n');
end

fprintf('All done.\n');
