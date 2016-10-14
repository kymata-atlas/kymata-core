
%NKG CBU QUEUE SCRIPT

% input arguments

%functionname = 'ins_loudness';
%stimulisigFunctionName = 'ins_loudnessSplit';
%inputstream = 'audio';
%functionlocation = ['/', inputstream, '/raw/GMloudness/'];
%inputfolder = ['vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone/', inputstream];
%outputfolder = 'vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone/with_etymotics_prefiltering';
inv = 'false';


functionname = 'brontesX';
stimulisigFunctionName = 'brontesXSplit';
inputstream = 'visual';
functionlocation = ['/', inputstream, '/raw/CIECAM02/'];
inputfolder = ['vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone/', inputstream];
outputfolder = 'vert10242-nodepth-diagonly-snr1-signed-fsaverage-baselineNone';

cutoff = 1000; 
nTimePoints = 2001;
nWords = 400;
nVertices = 10242;


%% ========================================================================
% specify various parameters
% list of directories that should be added to the matlab path on worker
% nodes. Worker nodes are fresh instances of matlab, and will start with 
% your default matlab path. If your job scripts are not on your default 
% path, include their directories below:

workerpath={'/imaging/at03/NKG_Code/Version6_dataset_1/CBU_QUEUE'};

%addpath('/opt/gold/bin')
%addpath('/cluster-software/gold/2.2.0.5/sbin')
%addpath('/cluster-software/maui/bin:/cluster-software/torque-2.3/sbin')
%addpath('/cluster-software/torque-2.3/bin:/opt/xcat/bin')
%addpath('/opt/xcat/sbin')
addpath('/hpc-software/matlab/cbu/','-BEGIN');
addpath(workerpath{:},'-BEGIN');

% list of tasks to run for each subject - for each subject, run the
% pre-processing, then the first level model.
tasks={'NKG_do_neurokymatography_standard'};
tasks=tasks(1);


%% ========================================================================
% Create a job array containing a list of all jobs the jobs you want to run
% in the format expected by the cbu_qsub function. 
                                     
njob=2;
    
handedness={'lh','rh'};

% initialise job structure:
clear jobs;
jobs(1:njob)=deal(struct('task',[],'n_return_values',[],'input_args',[]));

for i=1:njob
    jobs(i).task=str2func(tasks{1}); % create a function handle for the current task
    jobs(i).n_return_values=0;
    jobs(i).input_args={char(handedness(i)), functionname, functionlocation, stimulisigFunctionName, cutoff, nWords, nTimePoints, nVertices, outputfolder, inputfolder, inv};
end

%% ========================================================================
% Create a scheduler object. This allows Matlab to submit jobs to the
% Torque scheduler used on the CBU cluster.

clear scheduler;
scheduler=cbu_scheduler('custom',{'compute', 1, 94, 259200});  %Kymata -72 hours

%   , [' -N NKG_', functionname, '_', leftright]

%% ========================================================================
% Submit the jobs to the cbu cluster
cbu_qsub(jobs,scheduler,workerpath);


