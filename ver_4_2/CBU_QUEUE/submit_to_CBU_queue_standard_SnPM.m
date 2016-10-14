
%NKG CBU QUEUE SCRIPT

% input arguments
for i = 1:500
    leftright = 'lh';
    functionname = 'TS-pitch';
    functionlocation = '/TANDEM-STRAIGHT-F0/';
    stimulisigFunctionName = 'f0';
    isStandard = 1;
    cutoff = 600; 
    nTimePoints = 2068;
    nWords = 480;
    seed = i;

    % 
    % leftright = 'rh';
    % functionname = 'averaged_loudness_shortwindow';
    % functionlocation = '/Glasberg-Moore_Loudness/';
    % stimulisigFunctionName = 'loudnessaverage1Sones';
    % isStandard = 1;
    % cutoff = 1000; 
    % nTimePoints = 2068;
    % nWords = 480;

    if isStandard == 1
        nVertices = 2562; 
        outputfolder = 'vert2562-smooth5-nodepth-corFM-snr1-signed';
        inputfolder = 'vert2562-smooth5-nodepth-eliFM-snr1-signedwithminusequalszero';
    else
        nVertices = 10242;  
        outputfolder = 'averagemesh-vert10242-smooth5-corinvsol';
        inputfolder = 'vert10242-smooth5-nodepth-corFM-snr1-signedwithminusequalszero';
    end

    %% ========================================================================
    % specify various parameters
    % list of directories that should be added to the matlab path on worker
    % nodes. Worker nodes are fresh instances of matlab, and will start with 
    % your default matlab path. If your job scripts are not on your default 
    % path, include their directories below:


    workerpath={'/imaging/at03/NKG_Code/Version4_2/CBU_QUEUE'};

    addpath('/opt/gold/bin')
    addpath('/cluster-software/gold/2.2.0.5/sbin')
    addpath('/cluster-software/maui/bin:/cluster-software/torque-2.3/sbin')
    addpath('/cluster-software/torque-2.3/bin:/opt/xcat/bin')
    addpath('/opt/xcat/sbin')
    addpath('/hpc-software/matlab/cbu/','-BEGIN');
    addpath(workerpath{:},'-BEGIN');

    % list of tasks to run for each subject - for each subject, run the
    % pre-processing, then the first level model.
    tasks={'NKG_do_neurokymatography_standard_SnPM'};
    tasks=tasks(1);


    %% ========================================================================
    % Create a job array containing a list of all jobs the jobs you want to run
    % in the format expected by the cbu_qsub function. 

    njob=1;

    % initialise job structure:
    clear jobs;
    jobs(1:njob)=deal(struct('task',[],'n_return_values',[],'input_args',[]));

    jobs(1).task=str2func(tasks{1}); % create a function handle for the current task
    jobs(1).n_return_values=0;
    jobs(1).input_args={seed,leftright, functionname, functionlocation, stimulisigFunctionName, cutoff, nWords, nTimePoints, nVertices, outputfolder, inputfolder};


    %% ========================================================================
    % Create a scheduler object. This allows Matlab to submit jobs to the
    % Torque scheduler used on the CBU cluster.

    clear scheduler;
    if isStandard == 1
        scheduler=cbu_scheduler('custom',{'compute', 1, 47, 129599});   %Standard
    else
        scheduler=cbu_scheduler('custom',{'compute', 1, 94, 259200});  %Kymata -72 hours
    end

    %% ========================================================================
    % Submit the jobs to the cbu cluster
    cbu_qsub(jobs,scheduler,workerpath);
end

