%
% Runs ICA from the command line using EEGLAB's 'runica'. May be run on
%   Continuous or Epoched data. See 'help runica' for options.
%
% FORMAT:
%   [D cmd] = spm_eeglab_runica(S)
%
% INPUT: S (optional) structure with (optional) subfields: {defaults}
%   D	 - ['string'] filename of data {prompt}
%   chans    - [vector] channels to use {0 (use all D.channels.eeg)}
%   samppct  - [0-1] proportion of data to use {1}
%   excrej   - [1|0] exclude epochs marked for rejection? {0}
%   excbad   - [1|0] exclude channels marked as bad? {0}
%   newpath  - ['string'] new path for output file {none ('./')}
%   newpfx   - ['string'] new output file prefix {'ica_'}
%   args     - eeglab arguments, format: {'key1',value1,'key2',value2}
%   tmpfname - ['string'] if you want to save W and ICAsphere to a 
%                mat file (useful if matlab crashes before saving D)
%
%   - Supported/tested runica args:
%    'pca'      - [integer] num. principal dimensions to extract before ICA
%    'extended' - [1|0] use extended version of ICA (see 'help runica')
%    'maxsteps' - [integer] max num. of steps to compute
%
%   - Other options (try at your own risk: see 'help runica')
%    'sphering','weights','lrate','block','anneal','annealdeg',
%    'stop','bias','momentum','specgram','posact','ncomps',
%    'verbose','logfile'
%
% OUTPUT: [D cmd]
%   D data structure (also saved) with subfields:
%     ica.W	 - ICA weights
%     ica.sphere   - ICA sphering matrix
%     ica.(params) - (chans, samppct, userej)
%     ica.args     -
%   cmd - runica command used
%
% EXAMPLE:
%     S.D = 'my_raw_data.mat';
%     S.samppct = .7;
%     S.args = {'extended',1,'maxsteps',800,'pca',65}
%     [D cmd] = spm_eeglab_runica(S);
%
% By Jason Taylor (30/8/2007) as meg_runica.m
%  JT updated (25/2/2007) renamed spm_eeglab_runica, added 'args'
%  JT updated (28/4/2008) added 'excbad' to read & omit bad channels
%  RH/JT updated (29/4/2008) tmp file save optional; renamed sphere ICAsphere
%

function [D cmd] = spm_eeglab_runica_yh(S);


%-------------------------------------------------------
% Get Parameters:
%---------------

try   fname=S.D;
catch fname=spm_select(1,'mat','Select file for ICA...');
end

D=spm_eeg_ldata(fname);
[p fstem ext] = fileparts(fullfile(D.path,filesep,D.fname));
srate=D.Radc;
ndims=length(size(D.data));

try   chans=S.chans;
catch chans=0;
end
if chans==0
	chans=D.channels.eeg;
end

try   samppct=S.samppct;
catch samppct=1;
end

try   excrej=S.excrej;
catch excrej=0;
end

try   excbad=S.excbad;
catch excbad=0;
end

try
			 newpath=S.newpath;
			 ls('-d',newpath);
			 dopath=1;
catch
			 newpath='./';
			 dopath=0;
end

try   newpfx=S.newpfx;
catch newpfx='ica_';
end
newfstem=[newpfx fstem];

try   tmpfname=S.tmpfname;
      savetmp=1;
catch savetmp=0;
end

try   args=S.args
catch args='';
end

% Open EEGLAB (to load toolboxes, etc.), close gui figure:
tmp=which('eegplot');
if isempty(tmp)
	eeglab();
	eegfig=gcf; close(eegfig);
end


%-------------------------------------------------------
% Prepare runica command:
%-----------------------

%%% Get data and coerce into 2D matrix if necessary:

if ndims==3
	% Get epoched data:
	disp(sprintf('\n%s','Epoched data detected...'))

	% Reduce data (if samppct used):
	nepochs=size(D.data,3);
	% Shuffle 'em up:
	epindex=1:nepochs;
	eprand=randperm(nepochs);
	epfilt=eprand <= ceil(samppct*nepochs);

	% Omit rejected epochs (if flagged):
	if excrej==1
		 % reverse 0s & 1s:
		 rejfilt=~D.events.reject;
		 epfilt= epfilt & rejfilt;
	end
	usedata=epindex(epfilt);

	% Omit bad channels (if flagged):
	if excbad==1
		 chans=chans(chans~=D.channels.Bad)
	end

	disp(sprintf('%s%s%s\n','Using ',num2str(length(usedata)),' epochs.'))

	% Get data:
	data=D.data(chans,:,usedata);
	data=reshape(data,length(chans),size(data,2)*size(data,3));

elseif ndims==2
	% Get continuous data:
	sprintf('\n%s','Continuous data detected...')

	% Omit bad channels (if flagged):
	if excbad==1
		 chans=chans(chans~=D.channels.Bad)
	end

    % In case of blank space at beginning of file:
	samp1=find(D.data(1,:)~=0,1)+10;

	% Reduce data (if samppct used):
	sampn=D.Nsamples*samppct;
	usedata=samp1:sampn;

	% Get data:
	data=D.data(1:length(chans),usedata);

end %if ndims

clear D % to free up memory


%%%% Begin compiling command:

% Basic command:
cmd=( '[W,ICAsphere]=runica(data' );

% Arguments?:
% This hideous bit of code parses arguments passed in EEGLAB
%  style: {'key1','value1','key2','value2'}.
%  Maybe it deserves its own function one day...?
%
if sum(size(args))>0
	j=1;
	for i=1:length(args)/2
		 key=args{j};
		 val=args{j+1};
		 if isnumeric(val)
			  if length(val)==1
				   cmd=sprintf('%s,''%s'',%d',cmd,key,val);
			  else
				   cmd=sprintf('%s,''%s'',[%s]',cmd,key,num2str(val));
			  end
		 elseif ischar(val)
			  cmd=sprintf('%s,''%s'',''%s''',cmd,key,val);
		 elseif iscell(val)
			  x=val{:};
			  if isnumeric(x)
				   y='';
				   for i=1:size(x,1)
					    y=[y sprintf('%g ',x(i,:))];
					    if i<size(x,1)
						     y=[y '; '];
					    else
						     y=['{' y '}'];
					    end
				   end %for size x
				   cmd=sprintf('%s,''%s'',%s',cmd,key,y);
			  elseif isstr(x)
				   y='';
				   for i=1:size(val,1)
					    for j=1:size(val,2)
						     y=[y sprintf('''%s'' ', val{i,j})];
					    end
					    if i<size(val,1)
						     y=[y '; '];
					    else
						     y=['{' y '}'];
					    end
				   end %for i size val
				   cmd=sprintf('%s,''%s'',%s',cmd,key,y);
			  end %if isstr x
		 end %if iscell val
		 j=j+2;
	end %for i
	cmd=[cmd ');'];
else
	% (No defaults):
	cmd=[cmd ');'];
end


%%% Run command:

%[W,ICAsphere]=runica(data,'pca',pca,'extended',1,'maxsteps',800);

% Do runica:
eval(cmd);

% Save variables to temp file
if savetmp
	tmpfname=[newpath '/' tmpfname];
	save(tmpfname,'S','W','ICAsphere')
end


%%% STORE IN DATA STRUCTURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D=spm_eeg_ldata(fname);
D.ica.W=W;
D.ica.sphere=ICAsphere;
D.ica.chans=chans;
D.ica.samppct=samppct;
D.ica.args=args;

if dopath
	D.path=newpath;
end

eval(['! cp ' D.fnamedat ' ' [D.path '/' newfstem '.dat']]);
D.fname=[newfstem '.mat'];
D.fnamedat=[newfstem '.dat'];

save([D.path '/' D.fname],'D');


