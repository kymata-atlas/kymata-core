%---------------------------------------------
% This file is a pipe-line that takes a fif. file, triggers and
% model-signals and output the coherence between the MEG and this signal.
%---------------------------------------------

addpath /opt/neuromag/meg_pd_1.2/
addpath /imaging/at03/Fieldtrip/
addpath /opt/mne/matlab/toolbox/
addpath /imaging/at03/MNE-matlab-toolbox/
addpath /imaging/at03/Fieldtrip/fileio/
addpath /imaging/at03/NKG/sensor_space_code/


%---------------------------------------------
% Define variables
%---------------------------------------------

% specify filenames
participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];
%participentIDlist = [400];

% Options
prewindow  = 0.2;
postwindow = 0.4;
lpfreq     = 30;
folder     = 'lpfilter_30Hz';  % some description of the setting used here

%---------------------------------------------
% 1-preprocessed
%---------------------------------------------


for m = 1:length(participentIDlist)
    MEGdata1 = [];
    MEGdata2 = [];
    MEGdata3 = [];
    MEGdata4 = [];
    eventsdata1 = [];
    eventsdata2 = [];
    eventsdata3 = [];
    eventsdata4 = [];
    for part = 1:4
        MEGfilename = ['/imaging/at03/LexproMEG/meg08_0', num2str(participentIDlist(m)), '/meg08_0', num2str(participentIDlist(m)), '_part', num2str(part), '_raw_sss_movecomp_trd.fif'];
        eventlistFilename = ['/imaging/at03/LexproMEG/meg08_0', num2str(participentIDlist(m)), '/meg08_0', num2str(participentIDlist(m)), '_part', num2str(part), '-acceptedwordeventsFIELDTRIP.eve'];
        if(exist(MEGfilename, 'file'))

            %---------------------------------------------
            % create an empty data structure
            %---------------------------------------------

            cfg = [];
            cfg.dataset         = MEGfilename;
            cfg.hearderfile     = MEGfilename;
            cfg.continuous      = 'yes';

            cfg.trialdef.pre    = prewindow;
            cfg.trialdef.post   = postwindow;

            %---------------------------------------------
            % Import Fif file
            %---------------------------------------------

            % import amended eventfile
            fid = fopen(eventlistFilename);
            eventlist = textscan(fid, '%n %n %n %n %n %s');
            fclose(fid);
            fclose('all');

            events=[];
            for thisevent = 2:length(eventlist{1,5})

                %fill out 'events' which is used by
                %trialfun_lexpro and definetrial to locate the word
                events(thisevent-1,1).type = 'stimuli';
                events(thisevent-1,1).sample = eventlist{1,1}(thisevent, 1);                                              % the begining of the stimuli, in milliseconds
                events(thisevent-1,1).duration = ((eventlist{1,3}(thisevent, 1))-(eventlist{1,2}(thisevent, 1)))*1000;    % the duration of the stimuli, in milliseconds
                events(thisevent-1,1).value = eventlist{1,6}(thisevent, 1);                                               % the stimuli
                events(thisevent-1,1).offset = 0;                                                                         % the shift needed to map the audio to the
            end

            cfg.trialdef.events  = events;
            cfg.trialfun        = 'trialfun_LexPro';

            cfg = definetrial(cfg);

            %---------------------------------------------
            % Create pre-processed data, fo a single word
            %---------------------------------------------

            % lowpass, and baseline corrected, no artifact rejection as it has
            % been done manually

            cfg.channel    = {'MEG'};                            % read all MEG channels is {'MEG'}
            cfg.blc        = 'yes';
            cfg.blcwindow  = [-0.2 0];
            cfg.lpfilter   = 'yes';                              % apply lowpass filter
            cfg.lpfreq     = lpfreq;                             % in Hz (filtered already to ??)
            cfg.padding    = 0.5;                                % length to which the trials are padded for filtering

            eval(['MEGdata',  num2str(part) ,' = preprocessing(cfg);']);
            eval(['eventsdata', num2str(part) ,' = events;']);
        end
    end

    if(isempty(MEGdata4))
        outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/1-preprocessed/meg080', num2str(participentIDlist(m)), 'FULL-corr.mat'];
        outputeventsfilename = ['/imaging/at03/NKG/saved_data/', folder, '/1-preprocessed/meg080', num2str(participentIDlist(m)), '_eventsfull.mat'];


        thisfullparticipantMEG = ['meg_part_0' , num2str(participentIDlist(m))];
        cfg =[];
        %meg_part_0400 = appenddata(cfg, MEGdata1, MEGdata2, MEGdata3);
        eval([thisfullparticipantMEG '= appenddata(cfg, MEGdata1, MEGdata2, MEGdata3)']);
        eval(['save (outputfilename, ''', thisfullparticipantMEG, ''')']);

        %---------------------------------------------
        % Create master eventfile for 3
        %---------------------------------------------

        eventsfull = [eventsdata1; eventsdata2; eventsdata3];
        save (outputeventsfilename, 'eventsfull');

    else
        outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/1-preprocessed/meg080', num2str(participentIDlist(m)), 'FULL-corr.mat'];
        outputeventsfilename = ['/imaging/at03/NKG/saved_data/', folder, '/1-preprocessed/meg080', num2str(participentIDlist(m)), '_eventsfull.mat'];

        thisfullparticipantMEG = ['meg_part_0' , num2str(participentIDlist(m))];
        cfg =[];
        eval([thisfullparticipantMEG ' = appenddata(cfg, MEGdata1, MEGdata2, MEGdata3, MEGdata4)']);
        eval(['save (outputfilename, ''', thisfullparticipantMEG, ''')']);

        %---------------------------------------------
        % Create master eventfile for 4
        %---------------------------------------------

        eventsfull = [eventsdata1; eventsdata2; eventsdata3; eventsdata4];
        save (outputeventsfilename, 'eventsfull', '-v7.3');
    end
    %eval([ 'clear MEGdata1 MEGdata2 MEGdata3 MEGdata4 eventsdata1 eventsdata2 eventsdata3 eventsdata4 ', thisfullparticipantMEG ]);
end

%---------------------------------------------
% 2-timelocked
%---------------------------------------------

participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];

wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');
 
for m = 1:length(participentIDlist)
     inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/1-preprocessed/meg080', num2str(participentIDlist(m)), 'FULL-corr.mat'];
     inputeventfilename = ['/imaging/at03/NKG/saved_data/', folder, '/1-preprocessed/meg080', num2str(participentIDlist(m)), '_eventsfull.mat'];
     load (inputfilename);
     load (inputeventfilename);
     for thiswordpos = 1:length(wordlist{1,1})
        
         thiswordpositions = find (strcmp([eventsfull(:,1).value] , wordlist{1,1}{thiswordpos,1}));
         
         cfg =[];
         cfg.trials             = [thiswordpositions];
         cfg.vartrllength       = 2;  % or 0 (default) or 1. anoyingly the rounding is slightly of for some of the trials, but this seems to work

         if(~isempty(cfg.trials))
             AVEword = ['average_', wordlist{1,1}{thiswordpos,1},'_meg_part_0' , num2str(participentIDlist(m))];
             eval([   AVEword, ' = timelockanalysis(cfg, meg_part_0' , num2str(participentIDlist(m)), ')'          ]);
             outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/2-timelocked/meg080', num2str(participentIDlist(m)), '/', wordlist{1,1}{thiswordpos,1}, '.mat'];
             eval([   'save( outputfilename, ''', AVEword, ''')']);
         end
         eval([   'clear ', AVEword, ';'   ]);
     end
     eval([   'clear meg_part_0' ,  num2str(participentIDlist(m)), ';'   ]);
end

%---------------------------------------------
% 3-aligned
%---------------------------------------------

% clear all
% participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];
% 
% wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
% fid = fopen(wordlistFilename);
% wordlist = textscan(fid, '%s');
% fclose('all');
%   
% for m = 1:length(participentIDlist)
%     inputfilename = ['/imadging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/1-preprocessed/meg080', num2str(participentIDlist(m)), 'FULL-corr.mat'];
%     outputfilename = ['/imdaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/1-preprocessed/meg080', num2str(participentIDlist(m)), 'FULL-corr.mat'];
%     if (exists(inputfilename, 'file'))
%         
%         cfg             = [];
%         cfg.template    = [];
%         cfg.template(1..N) = datasets that are averaged into the standard
%         cfg.inwardshift = [];
%         
%         load (inputfilename);
%         AVEword = ['average_', wordlist{1,1}{thiswordpos,1},'_meg_part_0' , num2str(participentIDlist(m))];
%         GAVEword = ['grandaverage_', wordlist{1,1}{thiswordpos,1},'_meg_part_0' , num2str(participentIDlist(m))];
%         eval([  GAVEword = megrealign(cfg, ', AVEword, ')' ]);
% 
% eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
%         clear
%     end
% end

%---------------------------------------------
% 4-grandaverage (normal - no correcting by N00latencies (also, slow, so
% use faster version)) Suplanted by the one below (but test it's correct first)
%---------------------------------------------


% participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];

wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');



for thiswordpos = 1:length(wordlist{1,1}) % should be one
    inputsarray = [];
    for m = 1:length(participentIDlist)
        inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/2-timelocked/meg080', num2str(participentIDlist(m)), '/', wordlist{1,1}{thiswordpos,1}, '.mat'];
        if (exist(inputfilename, 'file'))
            load (inputfilename);
            AVEword = ['average_', wordlist{1,1}{thiswordpos,1},'_meg_part_0' , num2str(participentIDlist(m))];
            inputsarray = [inputsarray, ', ', AVEword];
        end
    end

    cfg=[];
    GAVEword = ['grandaverage_', wordlist{1,1}{thiswordpos,1}];
    eval([  GAVEword, ' = timelockgrandaverage(cfg ', inputsarray, ')' ]);
    outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/4-grandaverage/', wordlist{1,1}{thiswordpos,1}, '.mat'];
    eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
    clear average* gr*
    fclose('all');
end

%---------------------------------------------
% 4-grandaverage (Normalised by Ns100 where possible)
%---------------------------------------------


participentIDlist =               [320 323 324 327 348 350 366 371 377 380 397 400 401 402];
participentIDN100correctionlist = [110 115 115 100 100 120 103 115 125 95   95 130 117 110];

wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');


M = 306;
N = 1501;
 for thiswordpos = 1:length(wordlist{1,1})
     inputsarray = [];
     i = 0;
     temp = zeros(M, N);
     for m = 1:length(participentIDlist)
         inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/2-timelocked/meg080', num2str(participentIDlist(m)), '/', wordlist{1,1}{thiswordpos,1}, '.mat'];
         if (exist(inputfilename, 'file'))
             load (inputfilename);
             i = i+1;
             AVEword = ['average_', wordlist{1,1}{thiswordpos,1},'_meg_part_0' , num2str(participentIDlist(m))];
             eval([ 'shifted = circshift(', AVEword,' .avg, [0 -(', num2str(participentIDN100correctionlist(m)), '-100)]);' ]);
             temp(1:306,1:1501,i) = shifted;
         end
     end
 
     GAVEword = ['grandaverage_', wordlist{1,1}{thiswordpos,1}];
     eval([  GAVEword, ' = ', AVEword ]);
     eval([  GAVEword, '.avg = mean(temp, 3)' ]);
     outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/normalisedbyN100/4-grandaverage/', wordlist{1,1}{thiswordpos,1}, '.mat'];
     eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
     clear average* gr*
     fclose('all');
 end



%---------------------------------------------
% 5-grandaveragenovar
%---------------------------------------------


% participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];
% 
wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');

for i = 1:length(wordlist{1,1})
    inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/normalisedbyN100/4-grandaverage/', wordlist{1,1}{i,1}, '.mat'];
    disp(num2str(i));
    if (exist(inputfilename, 'file'))
        load (inputfilename);
        fclose('all');
        GAVEword   = ['grandaverage_', wordlist{1,1}{i,1} ];
eval([  GAVEword, ' = rmfield(', GAVEword, ',', '''var''', ');' ]);
eval([  GAVEword, ' = rmfield(', GAVEword, ',', '''cfg''', ');' ]);
        outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/normalisedbyN100/5-grandaveragenovar/', wordlist{1,1}{i,1}, '.mat'];
        eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
        clear gr*
        fclose('all');        
    end
end




%---------------------------------------------
% Construct RMRS and place in Grad positions
%---------------------------------------------

% participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];
% 
% wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
% fid = fopen(wordlistFilename);
% wordlist = textscan(fid, '%s');
% fclose('all');
% 
 inputsarray{400,1} = 0;
 for i = 1:length(wordlist{1,1})
   %  inputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/', folder, '/5-grandaveragenovar/', wordlist{1,1}{i,1}, '.mat'];
   %  disp(num2str(i));
     if (exist(inputfilename, 'file'))
         load (inputfilename);
         inputsarray{i,1} = wordlist{1,1}{i,1};
         fclose('all');
         GAVEword   = ['grandaverage_', wordlist{1,1}{i,1} ];
         eval([   'wordlength = length(',GAVEword,'.avg)' ]);
         for MEG = 3:3:306
                      eval([   GAVEword, '.avg(MEG,:) = sqrt(', GAVEword, '.avg(MEG-1, :).^2 +' GAVEword, '.avg(MEG-2, :).^2);' ]);
         end
         
         outputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/', folder, '/6-grandaveragenovarRMS/', wordlist{1,1}{i,1}, '.mat'];
         eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
         clear gr*
         fclose('all');        
     end
 end


%/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-
% END
%/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-


%---------------------------------------------
% *NEW* Script: Construct RMRS and place in Grad positions, before grand average (uses).
%---------------------------------------------
    
% RMSthenGA

participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];

wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');


%%RMS
for thiswordpos = 1:length(wordlist{1,1})  %should be one
    for m = 1:length(participentIDlist)
        inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/2-timelocked/meg080', num2str(participentIDlist(m)), '/', wordlist{1,1}{thiswordpos,1}, '.mat'];
        if (exist(inputfilename, 'file'))
            load (inputfilename);
            AVEword = ['average_', wordlist{1,1}{thiswordpos,1},'_meg_part_0' , num2str(participentIDlist(m))];
            eval([   'wordlength = length(', AVEword,'.avg);' ]);
            for MEG = 3:3:306
                eval([   AVEword, '.avg(MEG,:) = sqrt(', AVEword, '.avg(MEG-1, :).^2 +' AVEword, '.avg(MEG-2, :).^2);' ]);
            end

            outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/RMSthenGA/4-RMS/meg080', num2str(participentIDlist(m)), '/', wordlist{1,1}{thiswordpos,1}, '.mat'];
            eval([  'save( outputfilename, ''', AVEword, ''')' ]);
            fclose('all');
            clear average*;
        end
    end
end

%%then Grand average
 for thiswordpos = 1:length(wordlist{1,1})
     inputsarray = [];
     for m = 1:length(participentIDlist)
         inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/RMSthenGA/4-RMS/meg080', num2str(participentIDlist(m)), '/', wordlist{1,1}{thiswordpos,1}, '.mat'];
         if (exist(inputfilename, 'file'))
             load (inputfilename);
             AVEword = ['average_', wordlist{1,1}{thiswordpos,1},'_meg_part_0' , num2str(participentIDlist(m))];
             inputsarray = [inputsarray, ', ', AVEword];
         end
     end
 
     cfg=[];
     GAVEword = ['grandaverage_', wordlist{1,1}{thiswordpos,1}];
     eval([  GAVEword, ' = timelockgrandaverage(cfg ', inputsarray, ')' ]);
     outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/RMSthenGA/5-grandaverage/', wordlist{1,1}{thiswordpos,1}, '.mat'];
     eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
     clear average* gr*
     fclose('all');
 end

%%then no var

inputsarray{400,1} = 0;
 for i = 1:length(wordlist{1,1})
     inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/RMSthenGA/5-grandaverage/', wordlist{1,1}{i,1}, '.mat'];
     disp(num2str(i));
     if (exist(inputfilename, 'file'))
         load (inputfilename);
         inputsarray{i,1} = wordlist{1,1}{i,1};
         fclose('all');
         GAVEword   = ['grandaverage_', wordlist{1,1}{i,1} ];
 eval([  GAVEword, ' = rmfield(', GAVEword, ',', '''var''', ');' ]);
 eval([  GAVEword, ' = rmfield(', GAVEword, ',', '''cfg''', ');' ]);
         outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/RMSthenGA/6-RMSgrandaveragenovar/', wordlist{1,1}{i,1}, '.mat'];
         eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
         clear gr*
         fclose('all');        
     end
 end
 
 
%---------------------------------------------
% *NEW* Script: Construct RMRS and place in Grad positions, before grand
% average (uses). Manual Version -EXACTLY THE SAME AS GRAND AV IF USING MEAN
%---------------------------------------------
    
%%then Grand average
 for thiswordpos = 1:length(wordlist{1,1})
     inputsarray = [];
     i = 0;
     temp = zeros(306, 1501);
     for m = 1:length(participentIDlist)
         inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/RMSthenGA/4-RMS/meg080', num2str(participentIDlist(m)), '/', wordlist{1,1}{thiswordpos,1}, '.mat'];
         if (exist(inputfilename, 'file'))
             load (inputfilename);
             i = i+1;
             AVEword = ['average_', wordlist{1,1}{thiswordpos,1},'_meg_part_0' , num2str(participentIDlist(m))];
             eval([ 'temp(1:306,1:1501,i) = ', AVEword,' .avg;' ]);
         end
     end
 
     GAVEword = ['grandaverage_', wordlist{1,1}{thiswordpos,1}];
     eval([  GAVEword, ' = ', AVEword ]);
     eval([  GAVEword, '.avg = mean(temp, 3)' ]);
     outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/RMSthenGA/5-grandaverage-manual-mean/', wordlist{1,1}{thiswordpos,1}, '.mat'];
     eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
     clear average* gr*
     fclose('all');
 end
 
 %%then no var

 for i = 1:length(wordlist{1,1})
     inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/RMSthenGA/5-grandaverage-manual-mean/', wordlist{1,1}{i,1}, '.mat'];
     disp(num2str(i));
     if (exist(inputfilename, 'file'))
         load (inputfilename);
         fclose('all');
         GAVEword   = ['grandaverage_', wordlist{1,1}{i,1} ];
 eval([  GAVEword, ' = rmfield(', GAVEword, ',', '''var''', ');' ]);
 eval([  GAVEword, ' = rmfield(', GAVEword, ',', '''cfg''', ');' ]);
         outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/RMSthenGA/6-RMSgrandaveragenovar-manual-mean/', wordlist{1,1}{i,1}, '.mat'];
         eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
         clear gr*
         fclose('all');        
     end
 end
 
%---------------------------------------------
% 4-grandaverage *manual* (BEFORE RMS)
%---------------------------------------------


% participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];

wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');



for thiswordpos = 1:length(wordlist{1,1})
    inputsarray = [];
    i = 0;
    temp = zeros(306, 1501);
    for m = 1:length(participentIDlist)
        inputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/2-timelocked/meg080', num2str(participentIDlist(m)), '/', wordlist{1,1}{thiswordpos,1}, '.mat'];
        if (exist(inputfilename, 'file'))
            load (inputfilename);
             i = i+1;
             AVEword = ['average_', wordlist{1,1}{thiswordpos,1},'_meg_part_0' , num2str(participentIDlist(m))];
             eval([ 'temp(1:306,1:1501,i) = ', AVEword,' .avg;' ]);
        end
    end

    GAVEword = ['grandaverage_', wordlist{1,1}{thiswordpos,1}];
    eval([  GAVEword, ' = ', AVEword ]);
    eval([  GAVEword, '.avg = mean(temp, 3)' ]);
    outputfilename = ['/imaging/at03/NKG/saved_data/', folder, '/4-grandaverage-manual-mean/', wordlist{1,1}{thiswordpos,1}, '.mat'];
    eval([  'save( outputfilename, ''', GAVEword, ''')' ]);
    clear average* gr*
    fclose('all');
end
