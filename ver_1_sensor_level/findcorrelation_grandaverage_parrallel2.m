if matlabpool('size') == 0
    matlabpool open 5
end
 

%--------------------------
% Find corrolation
%--------------------------


addpath /opt/neuromag/meg_pd_1.2/
addpath /imaging/at03/Fieldtrip/
%addpath /opt/mne/matlab/toolbox/
addpath /imaging/at03/Fieldtrip/fileio/
addpath /imaging/at03/NKG/

% set variables

clear all

% parrallel settings 
%maxNumCompThreads = 7; 

%other variables
sensortype = 'GRANDAVE_then_RMS';       % 'GRANDAVE_then_MAGS' or 'GRANDAVE_then_RMS' or 'RMS_then_GRANDAVE'
window = 7;                             % milliseconds
epsilon = 0.00000001;                   % for rounding errors


% do stuff with variables
  
if (strcmp(sensortype, 'GRANDAVE_then_MAGS'))
    inputfolder = '5-grandaveragenovar';
elseif (strcmp(sensortype, 'RMS_then_GRANDAVE'))
    inputfolder = 'RMSthenGA/6-RMSgrandaveragenovar';
else
    inputfolder = '6-grandaveragenovarRMS';
end

%--------------------------
% Import and append signals
%--------------------------

participentIDlist = [320 323 324 327 348 350 363 366 371 372 377 380 397 400 401 402];

wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
%wordlistFilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/temp/meg3032-Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
fclose('all');
 
inputsarray{400,1} = 0;
for i = 1:length(wordlist{1,1})
    inputfilename = ['/imaging/at03/NKG/saved_data/no_filtering/', inputfolder, '/', wordlist{1,1}{i,1}, '.mat'];
    %inputfilename = ['/imaging/at03/NKG/saved_data/unittest/', inputfolder, '/', wordlist{1,1}{i,1}, '.mat'];
    disp(num2str(i));
    if (exist(inputfilename, 'file'))
        %load (inputfilename);
        inputsarray{i,1} = wordlist{1,1}{i,1};
        fclose('all');
     end
end
 
% Import stimuli order and signals for each participant

load /imaging/at03/NKG/signals/C0_deltaC0_deltadeltaC0/stimulisig.mat;

% Attatch true entropy dataset (lable: 'SIGtrue')

for i = 1:length(inputsarray)
        %Find word in stimuli for participant 1
        
        thisstimuliword = inputsarray{i,1};
        eval([ 'MEGWORD = grandaverage_', thisstimuliword ]);
        MEGWORD.label{307} = 'SIGtrue';
        thisstimuliwordposition = find (strcmp(thisstimuliword, stimulisig.name));
        for j=1:size(stimulisig.time, 2)
            if (stimulisig.time(thisstimuliwordposition, j) ~= 0 && stimulisig.time(thisstimuliwordposition, j) <= 400)
                thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
                thissig = stimulisig.f1(thisstimuliwordposition, j); %accoustic likelhood signal
                thistimeposinMEG = find(abs(thistime - MEGWORD.time(1,:)) < epsilon); %floating decimal point problem
                MEGWORD.avg(307, thistimeposinMEG) = thissig;
            end
        end
        eval([ 'grandaverage_', thisstimuliword, ' = MEGWORD' ]);
end

%generate random array, so that the baselines are never true 
randomperm = zeros(50,length(inputsarray));
for p=1:50
    bagofnumbers = randperm(length(inputsarray));
    if (bagofnumbers(length(bagofnumbers)) == length(bagofnumbers))
        bagofnumbers(length(bagofnumbers)) = bagofnumbers(1);
        bagofnumbers(1) = length(bagofnumbers);
    end
    for l=1:length(bagofnumbers)
        if (bagofnumbers(l) == l)
            temp = bagofnumbers(l+1);
            bagofnumbers(l+1) = bagofnumbers(l);
            bagofnumbers(l) = temp;
        end
        randomperm(p,l) = bagofnumbers(l);
    end
end

%Attatch jumbled entropy dataset (lable: 'SIGrandom')

for i = 1:length(inputsarray)
        thisstimuliword = inputsarray{i,1};
        eval([ 'MEGWORD = grandaverage_', thisstimuliword ]);
        for channel = 308:357
            ranpermnumber = channel-307;
            %Find word in stimuli for participant 1
            thisstimuliRANword = inputsarray(randomperm(ranpermnumber, i), 1);
            MEGWORD.label{channel} = ['SIGrand', num2str(ranpermnumber)];
            thisstimuliwordposition = find(strcmp(thisstimuliRANword, stimulisig.name));
            for j=1:size(stimulisig.time, 2)
                if (stimulisig.time(thisstimuliwordposition, j) ~= 0 && stimulisig.time(thisstimuliwordposition, j) <= 400)
                    thistime = (stimulisig.time(thisstimuliwordposition, j))/1000;
                    thissig = stimulisig.f1(thisstimuliwordposition, j);
                    thistimeposinMEG = find(abs(thistime - MEGWORD.time(1,:)) < epsilon); %floating decimal point problem
                    MEGWORD.avg(channel, thistimeposinMEG) = thissig;
                end
            end
        end
        eval([ 'grandaverage_', thisstimuliword, ' = MEGWORD;' ]);
end
 
 

% Replace zeros with NaNs
      
for i = 1:length(inputsarray) % for each word
    disp(num2str(i));
    thisstimuliword = inputsarray{i,1};
    eval([ 'MEGWORD = grandaverage_', thisstimuliword, ';' ]);
    for j=1:length(MEGWORD.time)
            disp([num2str(i), ':', num2str(j)]);
            MEGWORD.avg(MEGWORD.avg==0)=NaN;
    end
    eval([ 'grandaverage_', thisstimuliword, ' = MEGWORD' ]);
end

% % Low pass filter on MEGdata
% 
% for i = 1:length(inputsarray)
%         thisstimuliword = inputsarray{i,1};
%         eval([ 'MEGWORD = grandaverage_', thisstimuliword ]);
%         for j = 3:3:306
%             temp = conv(MEGWORD.avg(j,:),fir1(100,0.2));
%             temp2 = temp(:,51:end-50);
%             MEGWORD.avg(j,:) = temp2;
%         end
%         eval([ 'grandaverage_', thisstimuliword, ' = MEGWORD' ]);
% end

%Convert to single object, in same order as wordlist

for i = 1:length(inputsarray) % for each word
    
    thisstimuliword = inputsarray{i,1};
    wordstructure{1,i} = thisstimuliword;
    eval([ 'wordstructure{2,i} = grandaverage_', thisstimuliword, '.time(:,:);' ]);
    eval([ 'wordstructure{3,i} = grandaverage_', thisstimuliword, '.avg(:,:);' ]);
end

% -------------------------------
% Find correlation co-effients for both
% -------------------------------

outputpath = ['/home/at03/Named_Screenshots/pitch_praat/2-ztest_fishertrans__GR_thenrms_7window_nofilter_all_parcipents_pequals0.000001/'];
latencies = [-30:10:500];

truewords = zeros(306,length(wordstructure));
randwords = zeros(306,(length(wordstructure)*50));

AllMEGdata = wordstructure(3,:);
times = wordstructure(2,:);
words = wordstructure(1,:);

for q = 1:length(latencies);
    latency = latencies(q);
    permindex = 1;
    
    parfor i = 1:length(words) % for each word

        thisstimuliword = words{i};

        %-------------------
        % for the trueSig
        %-------------------
        trueSig = AllMEGdata{i}(307,:); % this is the same order as the word list
        trueSigtime = times{i};
        % crop
        trueSigtime = trueSigtime(~isnan(trueSig));
        trueSig = trueSig(~isnan(trueSig));


        truewordsensor = zeros(306,1);

        MEGdata = AllMEGdata{i};
        fulltimeline = times{i};

        disp([num2str(i)]);

        for j = 1:306

            averagechannelsig = zeros(1,length(trueSig));

            for k = 1:length(trueSigtime)
                channeltimepos = find (fulltimeline(1,:) == trueSigtime(k));
                % average = mean(hamming(window)' .* MEGWORD.avg(j,(channeltimepos-(ceil(window/2))+1):(channeltimepos-1+(ceil(window/2)))));
                average = mean(MEGdata(j,((channeltimepos+latency)-(ceil(window/2))+1):((channeltimepos+latency-1)+(ceil(window/2)))));
                averagechannelsig(k) = average;
            end

            % Plot it
            %scatter(averagechannelsig,keeptrueSig, 3), xlabel('averagechannelsig'), ylabel('keeptrueSig');

            % Record correlation: pearson's Rho ('corr2' tested against the 'corr' version, and it's the same)
            truewordsensor(j,1) = corr2(averagechannelsig, trueSig);

        end

        truewords(:,i) = truewordsensor;
    end



    for permnumber = 308:313  %357

        randwordblock = zeros(306,length(words));

        parfor i = 1:length(words)  % for each word

            thisstimuliword = words{i};


            %-------------------
            % for the randSig
            %-------------------
            randSig = AllMEGdata{i}(permnumber,:); % this is the same order as the word list
            randSigtime = times{i};
            % crop
            randSigtime = randSigtime(~isnan(randSig));
            randSig = randSig(~isnan(randSig));


            randwordsensor = zeros(306,1);

            MEGdata = AllMEGdata{i};
            fulltimeline = times{i};

            disp([num2str(i), ':', num2str(permnumber)]);

            averagechannelsig = zeros(1,length(randSigtime));

            for j = 1:306
                for k = 1:length(randSigtime)

                    channeltimepos = find(fulltimeline(1,:) == randSigtime(k));
                    average = mean(MEGdata(j,((channeltimepos+latency)-(ceil(window/2))+1):((channeltimepos+latency-1)+(ceil(window/2)))));
                    averagechannelsig(k) = average;
                end


                % Plot it
                %scatter(averagechannelsig,keeprandSig, 3), xlabel('averagechannelsig'), ylabel('keeprandSig');

                % Record correlation: pearson's Rho
                randwordsensor(j,1) = corr2(averagechannelsig, randSig);

            end

            randwordblock(:,i) = randwordsensor;
        end

        randwords(:,(permindex:permindex+length(words)-1)) = randwordblock;
        permindex = permindex + length(words);

    end

    %delete remaining zeroes
    randwords = reshape(randwords(randwords~=0),306,(length(randwords(randwords~=0)))/306);

    %-------------------------------
    % Distribution tests
    %-------------------------------

    %transform using Fisher's Z

    randwordsGuassian = zeros(size(randwords, 1), size(randwords, 2));
    truewordsGuassian = zeros(size(truewords, 1), size(truewords, 2));

    for word = 1:size(randwords, 2)
        for channel = 1:306
            r = randwords(channel,word);
            randwordsGuassian(channel,word) = 0.5*log((1+r)/(1-r));
        end
    end


    for word = 1:size(truewords, 2)
        for channel = 1:306
            r = truewords(channel,word);
            truewordsGuassian(channel,word) = 0.5*log((1+r)/(1-r));
        end
    end

    %to deal with positive and negitive
    for channel = 1:306
        if (mean(truewordsGuassian(channel,:)) < 0)
            truewordsGuassian(channel,:) = truewordsGuassian(channel,:).*-1;
        end
        if (mean(randwordsGuassian(channel,:)) < 0)
            randwordsGuassian(channel,:) = randwordsGuassian(channel,:)*-1;
        end
    end

    %lillefor test to check for Guassian

    %h = lillietest(randwordsGuassian(36,:));

    %histfit(randwordsGuassian(42,:), 30); % plot
    %hold on;
    %histfit(truewordsGuassian(42,:), 30);

    %h = findobj(gca,'Type', 'patch');
    %display(h);

    %set(h(1),'FaceColor',[0.982 0.909 0.721],'EdgeColor','k');
    %set(h(2),'FaceColor',[0.819 0.565 0.438],'EdgeColor','k');

    %-------------------------------
    % Do population test on signals
    %-------------------------------

    %make discete
    pvalues = [];
    sigvalue = 0.000001;
    %display(['The null hypothesis is that the specified channel does does not significantly encode TRUESIG information any more than it encodes for RANSIG.']);
    %display(['The significance value is ', num2str(sigvalue) ]);
    for channel = 1:306
        truepopulation = truewordsGuassian(channel, :);
        randpopulation  = randwordsGuassian(channel, :);
        [h,p,ci,zval] = ztest(truepopulation,mean(randpopulation),std(randpopulation), sigvalue, 'right');
        if(p<sigvalue)
           % display(['Channel ', num2str(channel), '. Null hypothesis for this channel rejected at p-value ', num2str(p) ,'*']);
        else
           % display(['Channel ', num2str(channel), '. Null hypothesis not rejected (p value:', num2str(p) , ')']);
        end
        pvalues = [pvalues p];
    end

    %---------------------------------
    % Prepare for topoplot
    %---------------------------------

    MAGpvalues = [];
    highlightchannels = [];
    i=1;
    for channel = 3:3:306
        MAGpvalues =[MAGpvalues ; pvalues(channel)];
        if (pvalues(channel)<sigvalue)
            highlightchannels = [highlightchannels i];
        end
        i=i+1;
    end

    datavector = MAGpvalues;

    %---------------------------------
    % Print in topoplot
    %---------------------------------


    col = pink(1024);    tmp = linspace(0,1,1024)';
    for n = 1:3, col(:,n) = interp1( 10.^tmp, col(:,n), 1+9*tmp, 'linear'); end
    colormap(col)

    cfg.colormap        = col;
    cfg.layout          = 'CBU_NM306mag.lay';
    cfg.colorbar        = 'WestOutside';        % outside left
    cfg.gridscale       = 140;                  % scaling grid size (default = 67)
    cfg.maplimits       = [0 1];                % Y-scale

    %cfg.style           = 'straight';           %(default)
    %cfg.contournum      = 6;                   %(default = 6), see CONTOUR
    %cfg.shading         = 'interp';             %(default = 'flat')

    cfg.style           = 'both';               %(default)
    cfg.contournum      = 1;                    %(default = 6), see CONTOUR
    cfg.shading         = 'flat';               %(default = 'flat')

    cfg.contcolor       = [57 23 91];
    cfg.interpolation   = 'v4';                 % default, see GRIDDATA
    cfg.electrodes      = 'highlights';         % should be 'highlights' for white dots. But also 'off','labels','numbers','highlights' or 'dotnum' (default = 'on')
    cfg.ecolor          = [1 1 1];              % Marker color (default = [0 0 0] (black))
    cfg.highlight       = highlightchannels;    % or the channel numbers you want to highlight (default = 'off'). These numbers should correspond with the channels in the data, not in the layout file.
    cfg.hlcolor         = [1 1 1];

    topoplot(cfg, datavector)
    title([num2str(latency), 'ms'], 'FontSize', 20, 'FontWeight','bold');
    sigstring = num2str(sigvalue, '%6.6f');
    outputfile = [outputpath, '/latency-', num2str(latency),'_pval-', sigstring, '.fig'];
    saveas(gcf, outputfile);


    
end