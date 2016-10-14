addpath /home/jan/public/fieldtrip
addpath /home/joachim/matlab/




data=allblockdata;

for k=1:100,  %look for last non-zeros value
    speechend=(find(allblockdata.trial{k}(end,end:-1:201)> 0));
    data.time{k}=data.time{k}(201:end-speechend(1));
    data.trial{k}=data.trial{k}(:,201:end-speechend(1));


end;

%cfg=[];
%cfg.method = 'summary';
%data = rejectvisual(cfg,data);

cfg = [];
cfg.output = 'powandcsd';
cfg.method = 'mtmfft';
cfg.taper = 'hanning';
cfg.foilim = [0 20];


%cfg.tapsmofrq = 2;
cfg.keeptrials = 'yes';
cfg.channel = {'MEG' 'ENV'};
cfg.channelcmb = {'MEG' 'ENV'};
freq = freqanalysis(cfg, data);
fre=freqdescriptives([],freq);




%now permute ENV across trials
% ri=randperm(100); %random permutation
% for k=1:100,
%     %adjust length of trial
%     tmp=data.trial{ri(k)}(end,:);
%     nsamp=length(data.time{k});
%     if (nsamp >= length(tmp)),


%         tmp=[tmp zeros(1,nsamp-length(tmp))];
%     else
%         tmp=tmp(1:nsamp);
%     end
%     data.trial{k}(end,:)=tmp;
% end;

%alternative: scramble each ENV time series
data2=data;

for k=1:100,

    %adjust length of trial
    tmp=data.trial{k}(end,:);
    ri=randperm(length(tmp));
    data2.trial{k}(end,:)=tmp(ri);
end;
cfg = [];
cfg.output = 'powandcsd';
cfg.method = 'mtmfft';


cfg.taper = 'hanning';
cfg.foilim = [0 20];
%cfg.tapsmofrq = 2;
cfg.keeptrials = 'yes';
cfg.channel = {'MEG' 'ENV'};
cfg.channelcmb = {'MEG' 'ENV'};
freq2 = freqanalysis(cfg, data2);


fre2=freqdescriptives([],freq2);
%[fd] = fdsem2fdT(fre, fre2,'cohspctrm',0,[],'equalvar');

%do some smooothing
for k1=1:306,
    fre.cohspctrm(k1,:)=filtfilt([1 2 1]/4,1,fre.cohspctrm(k1,:));


    fre2.cohspctrm(k1,:)=filtfilt([1 2 1]/4,1,fre2.cohspctrm(k1,:));
end;

%change label to match the layout file
for k=1:306,
    fre.label{k}=data.label{k}(4:end);
    fre.labelcmb{k}=data.label{k}(4:end);


    fre2.label{k}=data.label{k}(4:end);
    fre2.labelcmb{k}=data.label{k}(4:end);
end;
cfg                  = [];
cfg.layout           = 'NM306mag.lay';
cfg.showlabels       = 'yes';
cfg.interactive  = 'yes';


cfg.xlim = [2 20];
cfg.ylim = [0 1e-27];
%cfg.zparam  = 'cohspctrm';
%cfg.cohrefchannel = 'ENV';
multiplotER(cfg,fre,fre2);
