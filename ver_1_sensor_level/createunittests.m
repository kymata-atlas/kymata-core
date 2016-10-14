% inputfolder = '6-grandaveragenovarRMS';
%  
% wordlistFilename = ['/imaging/at03/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
% fid = fopen(wordlistFilename);
% wordlist = textscan(fid, '%s');
% fclose('all');
% 
% inputsarray{400,1} = 0;
% for i = 1:length(wordlist{1,1})
%     inputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/no_filtering/', inputfolder, '/', wordlist{1,1}{i,1}, '.mat'];
%     disp(num2str(i));
%      if (exist(inputfilename, 'file'))
%          load (inputfilename);
%          inputsarray{i,1} = wordlist{1,1}{i,1};
%          fclose('all');
%      end
% end
% 
% for i = 1:length(inputsarray)
%     thisstimuliword = inputsarray{i,1};
%     disp(num2str(i));
%     eval(['lengthG = length(grandaverage_', thisstimuliword, '.avg)']) ;
%     for j = 1:306
%         for k = 1:lengthG
%                 eval(['grandaverage_', thisstimuliword, '.avg(j, k) = (normrnd(1,0.2)-1)*10^(-12);']);
%         end
%     end
% end
% 
% 
% % place C0-signal in MEG3
% 
% load /imaging/at03/Method_to_locate_neurocorrelates_of_processes/signals/C0_deltaC0_deltadeltaC0/stimulisig.mat;

for i = 1:length(inputsarray)
    thisstimuliword = inputsarray{i,1};
    thisstimuliwordposition = find (strcmp(thisstimuliword, stimulisig.name));
    c0 = stimulisig.C0(thisstimuliwordposition,:);
    kron(c0,ones(1,2));
    c0(c0==0) = [];
    c0 = kron(c0,ones(1,10));
    %scale c0 to between -1 and 1, and scale it;
    c0 = c0./4000;
    c0 = c0.*(10^(-12));
    %add noise
    for j=1:length(c0)
        c0(j) = c0(j)+((normrnd(1,0.2)-1)*10^(-12));
    end
    for j=1:length(c0);
        thissig = c0(j); 
        thistimeposinMEG = j+199;
        eval(['grandaverage_', thisstimuliword, '.avg(42, thistimeposinMEG) = thissig;']);
    end
    word = ['grandaverage_', thisstimuliword];
    outputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/unittest/6-grandaveragenovarRMS/', thisstimuliword, '.mat'];
    eval([   'save( outputfilename, ''', word, ''')']);
end


inputsarray{400,1} = 0;
 for i = 1:length(wordlist{1,1})
     inputfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/saved_data/unittest/', inputfolder, '/', wordlist{1,1}{i,1}, '.mat'];
     disp(num2str(i));
      if (exist(inputfilename, 'file'))
          load (inputfilename);
          inputsarray{i,1} = wordlist{1,1}{i,1};
          fclose('all');
      end
 end