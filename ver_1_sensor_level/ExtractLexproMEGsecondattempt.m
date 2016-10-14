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
participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];

% Options
prewindow  = 0.2;
postwindow = 0.2;

% global variables
stimuli = [];


%---------------------------------------------
% 1-preprocessed
%---------------------------------------------

% MEGParticipantArray = [];
% 
% for m = 1:length(participentIDlist)
%     for part = 1:4
%         MEGfilename = ['/imaging/at03/LexproMEG/meg08_0', num2str(participentIDlist(m)), '/meg08_0', num2str(participentIDlist(m)), '_part', num2str(part), '_raw_sss_movecomp_tr.fif'];
%         eventlistFilename = ['/imaging/at03/LexproMEG/meg08_0', num2str(participentIDlist(m)), '/meg08_0', num2str(participentIDlist(m)), '_part', num2str(part), '-acceptedwordeventsFIELDTRIP.eve'];
% 
%         if(exist(MEGfilename))
% 
%             %---------------------------------------------
%             % create an empty data structure
%             %---------------------------------------------
% 
%             cfg = [];
%             cfg.dataset         = MEGfilename;
%             cfg.hearderfile     = MEGfilename;
%             cfg.continuous      = 'yes';
% 
%             cfg.trialdef.pre    = prewindow;
%             cfg.trialdef.post   = postwindow;
% 
%             %---------------------------------------------
%             % Import Fif file
%             %---------------------------------------------
% 
%             % import amended eventfile
%             fid = fopen(eventlistFilename);
%             eventlist = textscan(fid, '%n %n %n %n %s');
%             fclose(fid);
%             fclose('all');
% 
%             for thisevent = 2:length(eventlist{1,5})
%                 
%                 thisword = eventlist{1,5}{thisevent, 1};
%                 thisword2 = [thisword, num2str(2)]; %for repitition 
%                 
%                 %fill out 'events' which is used by
%                 %trialfun_lexpro and definetrial to locate the word
%                 events=[];
%                 events(1,1).type = 'stimuli';
%                 events(1,1).sample = eventlist{1,1}(thisevent, 1);              % the begining of the stimuli, in milliseconds
%                 events(1,1).duration = 0.5*1000;                                % the duration of the stimuli, in milliseconds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%                 events(1,1).value = thisword;                                   % the stimuli
%                 events(1,1).offset = 0;                                         % the shift needed to map the audio to the
% 
%                 cfg.trialdef.events  = events;
%                 cfg.trialfun        = 'trialfun_LexPro';
% 
%                 cfg = definetrial(cfg);
%                 
%                 %---------------------------------------------
%                 % Create pre-processed data, fo a single word
%                 %---------------------------------------------
% 
%                 % lowpass, and baseline corrected, no artifact rejection as it has
%                 % been done manually
% 
%                 cfg.channel    = {'MEG'};                            % read all MEG channels
%                 cfg.blc        = 'yes';
%                 cfg.blcwindow  = [-0.2 0];
%                 cfg.lpfilter   = 'yes';                              % apply lowpass filter
%                 cfg.lpfreq     = 55;                                 % in Hz (filtered already to??)
%                 cfg.padding    = 0.5;                                % length to which the trials are padded for filtering
%                 
%                 temp = preprocessing(cfg);
%     eval([      thisword ' = temp'                    ]);
%      
%                 outputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/1-preprocessed/meg080', num2str(participentIDlist(m)), '/part_', num2str(part),  '/', thisword ,'_data.mat'];
%                 outputfilename2 = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/1-preprocessed/meg080', num2str(participentIDlist(m)), '/part_', num2str(part),  '/', thisword ,'2_data.mat'];
%                 if(exist(outputfilename))
%     eval([          thisword2 ' = temp'                                 ]);
%     eval([          'save (outputfilename2, ''', thisword2, ''')'       ]);
%     
%     eval([          'clear ', thisword2                                 ]); 
%     eval([          'clear ', thisword                                  ]); 
%                 else
%     eval([          'save (outputfilename, ''', thisword, ''')'         ]);
%         
%     eval([          'clear ', thisword                                  ]); 
%                 end
%             end
%         end
%         clear eventlist, 
%     end
% end


%---------------------------------------------
% 2-mergedreps
%---------------------------------------------

%wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
%fid = fopen(wordlistFilename);
%wordlist = textscan(fid, '%s');
%fclose('all');

% merge reps in part_2 (402 works with this as well)

% for m = 1:length(participentIDlist)
%     for thiswordpos = 1:length(wordlist{1,1})        
%         thisword = wordlist{1,1}{thiswordpos,1};
%         thiswordfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/1-preprocessed/meg080', num2str(participentIDlist(m)), '/part_2/', thisword ,'_data.mat'];
%         thisword2filename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/1-preprocessed/meg080', num2str(participentIDlist(m)), '/part_2/', thisword ,'2_data.mat'];
%         if(exist(thisword2filename))
%             cfg =[];
% eval([      'load ', thiswordfilename                                        ]);
% eval([      'load ', thisword2filename                                       ]);
%             thisword2 = [thisword, num2str(2)]
% eval([      thisword, ' = appenddata(cfg, ', thisword, ', ', thisword2, ');' ]);
%             outputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/1-preprocessed/meg080', num2str(participentIDlist(m)), '/part_2/', thisword ,'1+2appended_data.mat'];
% eval([      'save (outputfilename, ''', thisword, ''');'                     ]);
% eval([      'clear ', thisword, thisword2, ';'                          ]);
%         end
%    end
% end



FLAG_1_exists

for m = 1:length(participentIDlist)
    FLAG_1_exists = 0;
    FLAG_2_exists = 0;
    FLAG_2b_exists = 0;
    FLAG_3_exists = 0;
    FLAG_4_exists = 0;
    MEGfilename1 = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(part), '_raw_sss_movecomp_tr.fif'];
    MEGfilename2 = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(part), '_raw_sss_movecomp_tr.fif'];
    MEGfilename3 = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(part), '_raw_sss_movecomp_tr.fif'];
    MEGfilename4 = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(part), '_raw_sss_movecomp_tr.fif'];
    MEGfilename2b = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/1-preprocessed/meg080', num2str(participentIDlist(m)), '/part_2/', thisword ,'1+2appended_data.mat'];
    
    
    %which file sexist?
    
    if(exist(MEGfilename1))
       FLAG_1_exists = 1;
    end
    if(exist(MEGfilename2))
       FLAG_2_exists = 1;
    end
    if(exist(MEGfilename3))
       FLAG_3_exists = 1;
    end
    if(exist(MEGfilename4))
       FLAG_4_exists = 1;
    end 
    if(exist(MEGfilename2b))
        FLAG_2b_exists = 0;
        FLAG_2b_exists = 1;
    end
    
    if (FLAG_1_exists)
        if (FLAG_2_exists)
            if (FLAG_3_exists)
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  1234
                else
                    load
                    load
                    load
                    load
                    appenddata(cfg,  123
                end
            else
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  124
                else
                    load
                    load
                    load
                    load
                    appenddata(cfg,  12
                end
            end
        elseif (FLAG_2b_exists)
            if (FLAG_3_exists)
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  12b34
                else
                    load
                    load
                    load
                    load
                    appenddata(cfg,  12b3
                end
            else
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  12b4
                else
                    load
                    load
                    load
                    load
                    appenddata(cfg,  12b
                end
            end
        else
            if (FLAG_3_exists)
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  134
                else
                    load
                    load
                    load
                    load
                    appenddata(cfg,  13
                end
            else
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  14
                else
                    save 1
                end
            end
        end
    else
        if (FLAG_2_exists)
            if (FLAG_3_exists)
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  234
                else
                    load
                    load
                    load
                    load
                    appenddata(cfg,  23
                end
            else
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  24
                else
                    load
                    load
                    load
                    load
                    Save 2
                end
            end
        elseif (FLAG_2b_exists)
            if (FLAG_3_exists)
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  2b34
                else
                    load
                    load
                    load
                    load
                    appenddata(cfg,  2b3
                end
            else
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg,  2b4
                else
                    load
                    load
                    load
                    load
                    save 2b
                end
            end
        else
            if (FLAG_3_exists)
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    appenddata(cfg, 34
                else
                    load
                    load
                    load
                    load
                    save 3
                end
            else
                if (FLAG_4_exists)
                    load
                    load
                    load
                    load
                    save 4
                else
                    % do nothing
                end
            end

            appenddata(cfg, xxx);

            save ['/imaging/at03/xxx/2-merged/meg08_03', num2str(participentIDlist(m)), '/' (0) 'data.mat'];
        end
    end

        
        
  
               save ['/imaging/at03/xxx/2-merged/meg08_03', num2str(participentIDlist(m)), '/' (0) 'data.mat'];
            MEGfilename = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(part), '_raw_sss_movecomp_tr.fif'];
            MEGfilename = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(part), '_raw_sss_movecomp_tr.fif'];
            MEGfilename = ['/imaging/at03/LexproMEG/meg08_03', num2str(participentIDlist(m)), '/meg08_03', num2str(participentIDlist(m)), '_part', num2str(part), '_raw_sss_movecomp_tr.fif'];
            appenddata(cfg, xxx);
            save ['/imaging/at03/xxx/2-merged/meg08_03', num2str(participentIDlist(m)), '/' (0) 'data.mat'];
end
          
%---------------------------------------------
% 3-timelocked
%---------------------------------------------

% for all files in xxx
%     avgAverage AVEword = timelockanalysis(cfg, thiswordpreproccessed);
%     save AVErageword ../participant/AVEword 
% end
                
%---------------------------------------------
% 4-aligned
%---------------------------------------------

% for all files in xxx
%     aling
%     save AVErageword ../participant/AVEword 
% end

%---------------------------------------------
% 5-grandaverage
%---------------------------------------------

% for all files in xxx
%     Grandavergae
%     save AVErageword ../participant/AVEword 
% end

%---------------------------------------------
% Construct planar gradients (analogous, but not the same as RMRS)
%---------------------------------------------

%planar gradients (GRADS?)


    