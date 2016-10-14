%-------------------------------------------
% Work out latency using cross correlation (needs all the grandaveraged word vaiables
% from grandaverage, the input list, or eqivilant)
%-------------------------------------------

sensorname      = 'MEG0111';
window          = [-20 1 100];  % start time, interval and endtime, 
permutations    = 10;
sigvalue        = 0.05;

% find channel
channel = find(stcpm(sensorname, grandaverage_bashed.channelname));

%global variables

latency.time  = [];
latency.pvalue  = [];

storetrue.lag = [];
storetrue.population = {};

storetrue.lag  = [];
storetrue.population = {};

% -------------------------------
% Find correlation co-effients for both
% -------------------------------

truewords = zeros(1,length(inputsarray));
randwords = zeros(1,(length(inputsarray)*permutations));

count=1;
for k = window(1):window(2):window(3)
    for i = 1:length(inputsarray) % for each word

        thisstimuliword = inputsarray{i,1};
        eval([ 'MEGWORD = grandaverage_', thisstimuliword, ';' ]);


        %-------------------
        % for the trueSig
        %-------------------
        trueSig = MEGWORD.avg(channel,:);
        % get rid of NaNs
        keepSigtime = MEGWORD.time((~isnan(MEGWORD.time(i))) & (~isnan(trueSig)));
        keeptrueSig = trueSig((~isnan(MEGWORD.time(i))) & (~isnan(trueSig)));

        averagechannelsig = [];
        for k = 1:length(keeptrueSig)
            channeltimepos = find (MEGWORD.time(1,:) == keepSigtime(k));
            average = mean(MEGWORD.avg(channel,(channeltimepos-window):(channeltimepos+window)));
            averagechannelsig(k) = average;
        end

        % Plot it
        % scatter(averagechannelsig,keeptrueSig, 3), xlabel('averagechannelsig'), ylabel('keeptrueSig');

        % Record correlation: pearson's Rho ('corr2' tested against the 'corr' version, and it's the same)
        truewords(1,i) = corr2(averagechannelsig, keeptrueSig);

    end
    storetrue.population{count} = truewords;
    storetrue.lag{count} = truewords;
    count = count+1;
end



for permnumber = 308:(307+permutations)
    for i = 1:length(inputsarray) % for each word

        thisstimuliword = inputsarray{i,1};
        eval([ 'MEGWORD = grandaverage_', thisstimuliword, ';' ]);


        %-------------------
        % for the randSig
        %-------------------
        randSig = MEGWORD.avg(permnumber,:);
        % get rid of NaNs
        keepSigtime = MEGWORD.time((~isnan(MEGWORD.time(i))) & (~isnan(randSig)));
        keeprandSig = randSig((~isnan(MEGWORD.time(i))) & (~isnan(randSig)));
        keeprandsigcopy  = keeprandSig;

        keeprandSig = keeprandsigcopy; % deleat as soon as you get the chance....
        disp([num2str(i), ':', num2str(channel), ':', num2str(permnumber)]);
        averagechannelsig = [];
        deleateelements = [];
        for k = 1:length(keeprandSig)
            channeltimepos = find(MEGWORD.time(1,:) == keepSigtime(k));
            if (channeltimepos+window > length(MEGWORD.time))
                keeprandSig(k:end)=[];

                break
            else
                average = mean(MEGWORD.avg(channel,(channeltimepos-window):(channeltimepos+window)));
                averagechannelsig(k) = average;
            end
        end


        % Plot it
        % scatter(averagechannelsig,keeprandSig, 3), xlabel('averagechannelsig'), ylabel('keeprandSig');

        % Record correlation: pearson's Rho
        index = i + (length(inputsarray) * (permnumber-308));
        randwords(1,index) = corr2(averagechannelsig, keeprandSig);

    end
end

%-------------------------------
% Do population test on signals
%-------------------------------

%make discete
time = window(1);
for x = length(storetrue.population)
    [h,p,ci,zval] = ztest(truewords(1, :), mean(randwords(1, :)), std(randwords(1, :)));
    latency.time(x) = time;
    [h,p,ci,zval] = ztest(truewords(1, :), mean(randwords(1, :)), std(randwords(1, :)));
    latency.pvalue(x) =  p;
    time = time + window(2);
end


latency.time =   [-10   -9    -8    -7  -6   -5   -4   -3   -2   -1   0    1    2    3    4    5    6    7    8    9    10    11    12   13    14    15   16   17   18   19   20   21    22   23   24   25   26   27   28    29   30 ];
latency.pvalue = [0.987 0.937 0.74 0.86 0.84 0.75 0.96 0.97 0.76 0.91 0.78 0.45 0.05 0.67 0.90 0.93 0.94 0.74 0.97 0.95 0.90  0.93  0.94 0.77  0.97  0.95 0.90 0.93 0.94 0.77 0.97 0.85 0.90  0.83 0.94 0.87 0.97 0.95 0.90  0.93 0.94];

%now plot

lagplot = plot(latency.time,latency.pvalue);
xlabel('Potential latency of process in millisecond');
ylabel('p-value');
title('P-values for shifted signals (below zero is shown for completeness)');
set(lagplot,'Color','k','LineWidth',2);
display(lagplot);
hold all;
plot([-10 30],[0.05 0.05], '-.k');

