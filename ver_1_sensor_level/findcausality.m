addpath /imaging/at03/thirdparty_matlab_code/granger/

% % -------------------------------
% % Find Granger-causality for both
% % -------------------------------

window = 3;
%causality
alpha = 0.005;        %   alpha -- the significance level specified by the user
max_lag = 10;        %   max_lag -- the maximum number of lags to be considered

truewords = zeros(306,length(inputsarray));
randwords = zeros(306,(length(inputsarray)*50));

for i = 1:length(inputsarray) % for each word

     thisstimuliword = inputsarray{i,1};
     eval([ 'MEGWORD = grandaverage_', thisstimuliword, ';' ]);
     
    
    %-------------------
    % for the trueSig
    %-------------------
    trueSig = MEGWORD.avg(307,:);
    % get rid of NaNs
    keepSigtime = MEGWORD.time((~isnan(MEGWORD.time(i))) & (~isnan(trueSig)));
    keeptrueSig = trueSig((~isnan(MEGWORD.time(i))) & (~isnan(trueSig)));


    for j = 1:306
        disp([num2str(i), ':', num2str(j)]);
        averagechannelsig = [];
        for k = 1:length(keeptrueSig)
            channeltimepos = find (MEGWORD.time(1,:) == keepSigtime(k));
            average = mean(MEGWORD.avg(j,(channeltimepos-window):(channeltimepos+window)));
            averagechannelsig(k) = average;
        end

        % Plot it
        % scatter(averagechannelsig,keeptrueSig, 3), xlabel('averagechannelsig'), ylabel('keeptrueSig');

        % Record granger causality
        [F,c_v] = granger_cause([averagechannelsig],[keeptrueSig],alpha,max_lag);
      
        truewords(j,i) = F;
        
    end
end


for permnumber = 308:357  
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
        
        for j = 1:306
            keeprandSig = keeprandsigcopy; % deleat as soon as you get the chance....
            disp([num2str(i), ':', num2str(j), ':', num2str(permnumber)]);
            averagechannelsig = [];
            for k = 1:length(keeprandSig)
                channeltimepos = find(MEGWORD.time(1,:) == keepSigtime(k));
                if (channeltimepos+window > length(MEGWORD.time))
                    keeprandSig(k:end)=[];
                    
                    break
                else
                    average = mean(MEGWORD.avg(j,(channeltimepos-window):(channeltimepos+window)));
                    averagechannelsig(k) = average;
                end
            end
            

            % Plot it
            % scatter(averagechannelsig,keeprandSig, 3), xlabel('averagechannelsig'), ylabel('keeprandSig');

            % Record correlation: pearson's Rho
            index = i + (length(inputsarray) * (permnumber-308));
            randwords(j,index) = corr2(averagechannelsig, keeprandSig);

            [F,c_v] = granger_cause([averagechannelsig],[keeprandSig],alpha,max_lag);
      
            randwords(j,index) = F;
            
        end
    end
end

