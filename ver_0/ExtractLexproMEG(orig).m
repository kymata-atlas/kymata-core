%---------------------------------------------
% This file is a pipe-line that takes a fif. file, triggers and
% model-signals and output the coherence between the MEG and this signal.
%---------------------------------------------

addpath /opt/neuromag/meg_pd_1.2/
addpath /imaging/at03/Fieldtrip/
addpath /opt/mne/matlab/toolbox/
addpath /imaging/at03/Fieldtrip/fileio/


%---------------------------------------------
% Define variables
%---------------------------------------------

% specify filenames
participentIDlist = [19 20 23 24 27];
stimuliIDfilename = '/imaging/at03/LexproMEG/Simuli-Lexpro-MEG.txt';
modelSignalfilename = '/imaging/at03/LexproMEG/code.txt';

% % Options
% prewindow  = 0.2;
% postwindow = 0.5;
% 
% % global variables
% stimuli = [];
% 
% 
% %---------------------------------------------
% % Import Fif files (max filtered already, no downsampling)
% %---------------------------------------------
% 
% MEGParticipantArray = [];
% 
% for m = 1:length(participentIDlist)
% 
%     % create an empty data structure
% 
%     MEGtrialdata = [];
%     acceptedstimuli = {};
%     
%     for q = 1:3  %WHAT ABOUT 4 IN THE LAST ONE???
%         
%         MEGfilename = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(q), '_raw_sss_movecomp_tr.fif'];
%         eventlistFilename = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(q), '-cor.eve'];
%         uselistFilename = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(q), '-use.eve'];
% 
%         
%         cfg = [];
%         cfg.dataset         = MEGfilename;
%         cfg.hearderfile     = MEGfilename;
%         cfg.continuous      = 'yes';
% 
%         cfg.trialdef.pre    = prewindow;
%         cfg.trialdef.post   = postwindow;
%         
%         %---------------------------------------------
%         % Split the data into trials
%         %---------------------------------------------
%           
%         % import Elizabeths events as <nx3> matrix (could do it straight
%         % out of the Neuormag file, but this is less complicated)
%         eventlist = load(eventlistFilename);
%         uselist = load(uselistFilename);
%         
%         % import stimuil array
%         fid = fopen(stimuliIDfilename);
%             stimuliIDfile = textscan(fid, '%s %s %s %s %s %s %s %s');
%         fclose(fid);
%    
%     
%         % Mesh the two together as a new event array, and also add to the
%         % translator array
%         
%         events = [];        
%         j=1;
%         for i=1:3:length(eventlist)
%             % Reject for artifacts. check in 'use'.eve
%             if (isequal(find(uselist == eventlist(i)),zeros(0,1)))
%                 display(['rejecting sample starting at ' num2str(eventlist(i))]);
%             else
%                 display(['accepting sample starting at ' num2str(eventlist(i))]);
%                 events(j,1).type = 'stimuli';
%                 events(j,1).sample = eventlist(i);
%                 events(j,1).duration = eventlist(i+1) - eventlist(i);         % the duration of the stimuli
%                 % deal with the "second presentation" i.e. the point were each
%                 % number ends in a two not a one
%                 if rem(eventlist(i,4),2) ~= 0 % i.e. if odd
%                     temp = stimuliIDfile{1,(eventlist(i,4)-103)/10};
%                 else
%                     temp = stimuliIDfile{1,(eventlist(i,4)-100)/10};
%                 end
%                 events(j,1).value = temp{eventlist(i+2,4),1};                 % the stimuli
%                 events(j,1).offset = 0;                                       % the shift needed to map the audio to the
%                         
% 
% 
%     %outputfilename = ['/imaging/at03/Fieldtrip_recogniser_coherence/saved_data/MEGpreprocessed_participant_0803', num2str(participentIDlist(m)), '-corr.mat'];
%     %eval(['save (outputfilename,', h]);
%     %clear h;
%                 acceptedstimuli = [acceptedstimuli temp{eventlist(i+2,4),1}] ; 
%                 j = j+1;
%             end
%         end
%         
%         cfg.trialdef.events  = events;
%         cfg.trialfun        = 'trialfun_LexPro'; 
%         
%         cfg = definetrial(cfg);
%         
%         % Check triggers are in the correct place
%         %
%         
%         
%         %---------------------------------------------
%         % Create pre-processed data
%         %---------------------------------------------
%         
%         % lowpass, and baseline corrected, no artifact rejection as it has
%         % been done manually
% 
%         cfg.channel    = {'MEG'};                            % read all MEG channels
%         cfg.blc        = 'yes';
%         cfg.blcwindow  = [-0.2 0];
%         cfg.lpfilter   = 'yes';                              % apply lowpass filter
%         cfg.lpfreq     = 55;                                 % in Hz (Should be 55)
%         cfg.padding    = 0.5;                                % length to which the trials are padded for filtering
% 
%         v = ['meg_' , num2str(q)];
%         
%         eval([v ' = preprocessing(cfg)']);
%         
%         outputfilename = ['/imaging/at03/Fieldtrip_recogniser_coherence/saved_data/MEGpreprocessed_participant_0803', num2str(participentIDlist(m)), '_part', num2str(q),  '-corr.mat'];
%         eval(['save (outputfilename, ''', v, ''')']);
%         
%         eval(['clear ', v]); 
% 
%     end 
%     
%     for i=1:length(acceptedstimuli)
%         stimuli.values{m,1}{1,i} = acceptedstimuli{i};
%     end
%     stimuli.labels{m,1} = num2str(participentIDlist(m));
%    
%     clear meg_1  meg_2  meg_2;
%    
% end
% 
% save '/imaging/at03/Fieldtrip_recogniser_coherence/saved_data/stimuli.mat' stimuli;
% 

%----------------------------
% Append the parts together
%----------------------------

for m = 3:length(participentIDlist)

    for q = 1:3  %WHAT ABOUT 4 IN THE LAST ONE???
        
        inputfilename = ['/imaging/at03/Fieldtrip_recogniser_coherence/saved_data/MEGpreprocessed_participant_0803', num2str(participentIDlist(m)), '_part', num2str(q),  '-corr.mat'];
        eval(['load ', inputfilename]);
    
    end
    h = ['meg_part' , num2str(participentIDlist(m))];
    cfg =[];
    eval([h ' = appenddata(cfg, meg_1, meg_2, meg_3)']);
    outputfilename = ['/imaging/at03/Fieldtrip_recogniser_coherence/saved_data/MEGpreprocessed_participant_0803', num2str(participentIDlist(m)), 'FULL-corr.mat'];
    eval(['save (outputfilename, ''', h, ''')']);
    
    eval(['clear', h, 'meg_1 meg_2 meg_3']);
    
end

    