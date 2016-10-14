%
% Function to plot SPM MEEG data (continuous or epoched) using 
%   EEGlab's eegplot (see 'help eegplot' and/or 'help eeglab')
%
%   Notes: EEGlab toolbox must be in path.
%          Epochs previously marked for rejection are colored red.
%          To interactively mark epochs, try spm_eeg_rejectcnt.
%
%------------------------------------------------------------------
%
% FORMAT: [fig,chfig,data] = spm_eeg_plotcnt_child(S)
%
% INPUT: S (optional) structure with (optional) subfields {defaults}:
%   D        - filename of SPM MEEG data {prompt}
%   altdata  - use alternative data matrix, not D.data {0 (use D.data)}
%              may also specify {'acts'} for ICA activations
%   D2       - name of second dataset to overlay in red {0 (none)}
%   altdata2 - use alternative data2 matrix, not D2.data {0 (use D2.data)}
%   chans    - channels to plot {D.channels.eeg (all)}
%   chlabs   - channel labels (0 (#s) | 1 (names) | {list}) {0}
%
% -eegplot format options:
%   twin     - time window (or epochs) to plot {[start end]}
%   dispch   - number of channels to display at once {20}
%   wlen     - length of window, in sec or epochs {10}
%   spc      - spacing or scaling of amplitudes {500}
%   args     - eeglab arguments in {'keyword','value'} pairs {0 (none)}
%
% -child plot options:
%   child    - data to plot in 'child' window  {'off' (none)} 
%              Options are: 
%              {'ch#'} | {'ic#'} | {'veog'} | {'heog'} | {'blink'} | {'ecg'}
%              Where # is the index of a channel (with 'ch'), or
%              an independent component (with 'ic'); 
%              'blink' is index in D.ica.class.blink (same: 'pulse','horiz')
%              may combine, e.g. {'veog' 'heog' 'ic1' 'ic12' 'ch2'};
%              add + to flip IC activation (neg->pos or v.v.) e.g. {'ic2+'}
%              
% OUTPUT:
%   fig     - figure handle of main plot window
%   chfig   - 
%   data    - data plotted (useful if altdata=={'acts'};
%
%------------------------------------------------------------------
% Created by Jason Taylor (4/10/2007) as meg_display_cnt
%  JT updated (10/12/2007) to add D2
%  JT updated (25/1/2008) renamed spm_eeg_plotcnt
%  JT updated (6/2/2008) added 'child' plot option
%  JT updated (7/2/2008) added 'twin' option [not yet functional]
%                              'altdata', 'altdata2' options
%                              'args' option [not yet functional]
%  JT updated (2/3/2009) made 'altdata' functional
%                        added 'acts' option
%------------------------------------------------------------------
%

% FUTURE OPTIONS(?):
%  - load buffers instead of full dataset?
%  - select montages?
%  - interactive filters etc?

function [fig,chfig,data] = meg_plotcnt_caro(S)

%%% Parameters %%%

try    tmp=S.D;
		   if isstruct(tmp)
		   	fname=tmp.fname;
			 else
			 	fname=tmp;
			 end
catch  fname=spm_select(1,'any','Select data file to display');
end
D=spm_eeg_ldata(fname);
srate=D.Radc;
ndims=length(size(D.data));

try    altdata=S.altdata;
catch  altdata=[];
end
if ~isempty(altdata)
	if iscell(altdata) | isstr(altdata)
		if strcmpi(altdata,'acts')
			disp(sprintf('Computing IC activations...'))
			clear A
			A.D=fname;
			A.samppct=1;
			data=spm_eeglab_icaact(A);
		else
			disp(sprintf('Alternative data %s not understood; plotting MEG data',char(altdata)))
		end
	else
		data=altdata;
	end
end

if isempty(altdata)
	try    chans=S.chans;
	catch  chans=D.channels.eeg;
	end
	nchan=length(chans);
else
	nchan=size(data,1);
	chans=1:nchan;
end

try	   chlabs=S.chlabs;
catch  chlabs=0;
end

try    twin=S.twin;
catch  twin=0;
end

try    fname2=S.D2; 
       D2=spm_eeg_ldata(fname2); 
       dsets=2;
catch  dsets=1;
	     d2cmd='';
end

try    altdata2=S.altdata2;
catch  altdata2=[];
end
if ~isempty(altdata2)
	if iscell(altdata2) | isstr(altdata2)
		if strcmpi(altdata2,'acts')
			disp(sprintf('Computing IC activations...'))
			clear A
			A.D=fname2;
			A.samppct=1;
			data2=spm_eeglab_icaact(A);
		else
			disp(sprintf('Alternative data2 %s not understood; plotting MEG	data',char(altdat2a)))
		end
	else
		data2=altdata2;
	end
end

try 	 dispch=S.dispch;
catch  dispch=20;
end

try    wlen=S.wlen;
catch  wlen=10;
end

try    spc=S.spc;
catch  spc=500;
end

try    child=S.child;
catch  child={'off'};
end
if ~iscell(child)
	child=cellstr(child);
end

%%% Processing %%%
% Open EEGLAB (to load toolboxes, etc.)
tmp=which('eegplot'); 
if isempty(tmp)
	eeglab();
	close(findobj(0,'Tag','EEGLAB')); % close EEGLAB window
end

% Child-window stuff:
pc=[];pctype=[];doact=0;icmult=[];

if ~strcmpi(child{1},'off')
	for i = 1:length(child)
		switch lower(child{i}(1:2))
		case {'ve' 'he' 'ec'} % -------EOG (type 1)
			pctype=[pctype 1];
			if strcmpi(child{i},'veog')
				pc=[pc D.channels.veog];
				child{i}=['CH' num2str(pc(i)) ':VEOG'];
			elseif strcmpi(child{i},'heog')
				pc=[pc D.channels.heog];
				child{i}=['CH' num2str(pc(i)) ':HEOG'];
			elseif strcmpi(child{i},'ecg')
				pc=[pc D.channels.ecg];
				child{i}=['CH' num2str(pc(i)) ':ECG'];
			end
		case {'ch'} % -------------CHANNEL (type 2)
			pctype=[pctype 2];
			pc=[pc str2num(child{i}(3:end))];
			child{i}=['CH' num2str(pc(i)) ':' D.channels.name(pc(i))];
		case {'ic' 'bl' 'pu' 'ho'} % ---ICA component (type 3) *** VECTORS? ***
			pctype=[pctype 3];
			doact=1;
			if strcmpi(child{i}(end),'+')
				icmult=[icmult -1];
				ind=length(child{i})-1;
			else
				icmult=[icmult 1];
				ind=length(child{i});
			end				
			if strcmpi(child{i}(1:2),'ic')
				ic=str2num(child{i}(3:ind));
				child{i}=['IC' num2str(ic)];
			elseif strcmpi(child{i}(1:5),'blink')
				ic=D.ica.class.blink;
				child{i}=['IC' num2str(ic) ':blink'];
			elseif strcmpi(child{i}(1:5),'pulse')
				ic=D.ica.class.pulse;
				child{i}=['IC' num2str(ic) ':pulse'];
			elseif strcmpi(child{i}(1:5),'horiz')
				ic=D.ica.class.horiz;
				child{i}=['IC' num2str(ic) ':horiz'];
			end
			pc=[pc ic];
		end % switch
	end % for i in child
	childcmd='''children'',childfig,''command'',''close(findobj("tag","childEEG"))'',';
else
	childcmd='';
end % child ~ 'off'
% Report:
%[num2cell(pc') num2cell(pctype') child']


%*** FIX THIS!! *******
% Get time window in samples:
if twin==0
	
else
	twin=twin(1):twin(end);

end

%**********************

% Channel Labels:

chindex=1:length(chans);
chindex=sort(chindex,2,'descend'); % must reverse order (?)
if chlabs==0
	% Numbers:
	chnames=chans(chindex);
	chnames=cellstr(num2str(chnames'));
	chnames=char([' ';chnames]);
else
	% Labels (names):
	chnames=D.channels.name(chans);
	chnames=chnames(chindex);
	chnames=[' ';chnames]; % first entry must be blank (?)
	chnames=char(chnames);
end


%%% Get data:

% Get activations only once (if necessary):
if doact==1
	A.D=fname; A.doplot=0; A.userej=0; A.positive=0;
	acts=meg_icaact(A);
	acts=acts(pc(find(pctype==3)),:);
end

if ndims==3  % Epoched data

	nsamp=D.Nsamples;
	nepochs=D.Nevents;

	% Main dataset:
	if isempty(altdata)
		data=D.data(chans,:,:);
		ptitle=['Epoched Data: ' fname];
	elseif iscell(altdata) | isstr(altdata)
		if strcmpi(altdata,'acts')
			% (data (acts) computed above)
			ptitle=['IC Activations of Epoched Data: ' fname];
		end
	end

	% (optional) Second dataset:
	if dsets==2
		if isempty(altdata2)
			data2=D2.data(chans,:,:);
			d2cmd='''data2'',data2,';
			ptitle=[ptitle ' & ' fname2];
		elseif iscell(altdata2) | isstr(altdata2)
			if strcmpi(altdata2,'acts')
				% (data2 (acts) computed above)
				d2cmd='''data2'',data2,';
				ptitle=[ptitle ' & ' fname2];
			end
		end
	end

	
	% (optional) Child-window data:
	if ~strcmpi(child{1},'off')
		pclist=sprintf('%s, ',child{:});
		pcdata=[];iccount=0;
		for i=1:length(child)
			if ismember(pctype(i),[1 2])  %-- EOG or Channel data
				cdata=D.data(pc(i),:,:);
				cdata=reshape(cdata,1,nsamp,nepochs);
			elseif pctype(i)==3  %----------- IC activation
				iccount=iccount+1;
				cdata=icmult(iccount)*acts(iccount,:);
				cdata=reshape(cdata,1,nsamp,nepochs);
			end
			pcdata=cat(1,pcdata,cdata);
			clear cdata
		end % for i = child
		cdispch=size(pcdata,1);
		% Hack: eegplot doesn't recognise epochs if plotting single channel
		if cdispch==1
			pcdata=cat(1,pcdata,zeros(size(pcdata))); % hack!
			child=[child '(ignore)'];
		end
		% Normalise if more than one data type:
		pctypes=unique(pctype);
		if length(pctypes)>1
			ctitle=['Epoched data: ' pclist ' magnitude normalised by data type'];
			cspc=2;
			for i=1:length(pctypes)
				t=find(pctype==pctypes(i));
				pcdata(t,:,:)=pcdata(t,:,:)/max(max(max(abs(pcdata(t,:,:)))));
			end
		else
			ctitle=['Epoched data: ' pclist ' real magnitude'];
			if pctypes==1      % EOG
				cspc=500;
			elseif pctypes==2  % CH
				if isempty(altdata) % ie., main window is MEG data too
					cspc=spc;
				else
					cspc=size(pcdata,1)*max(max(max(abs(pcdata))));					
				end
			elseif pctypes==3  % IC
				cspc=5;
			end
		end
	end % ~ child 'off'

	% Convert old rejected trials into sample indices:
	sampindex=[0:nsamp:nsamp*nepochs];
	oldrej=D.events.reject;
	oldrejtr=find(oldrej>0);
	oldrejsamp=sampindex(oldrejtr)';

	% For EEGLAB to show rejected trials:
	oldvec=[1 .9 .9 zeros(size(chans))];
	wrej=[];
	for i=1:length(oldrejsamp)
		wrej(i,:)=[oldrejsamp(i) oldrejsamp(i)+nsamp oldvec];
	end
	wrejcmd='''winrej'',wrej,''wincolor'',[1 .7 .7],';

elseif ndims==2  	% Continuous data

	% Main dataset:
	if isempty(altdata)
		data=D.data(chans,:);
		ptitle=['Continuous Data: ' fname];
	elseif iscell(altdata) | isstr(altdata)
		if strcmpi(altdata,'acts')
			% (data (acts) computed above)
			ptitle=['IC Activations of Continuous Data: ' fname];
		end
	end

	% (optional) Second dataset:
	if dsets==2
		if isempty(altdata2)
			data2=D2.data(chans,:);
			d2cmd='''data2'',data2,';
			ptitle=[ptitle ' & ' fname2];
		elseif iscell(altdata2) | isstr(altdata2)
			if strcmpi(altdata2,'acts')
				% (data2 (acts) computed above)
				d2cmd='''data2'',data2,';
				ptitle=[ptitle ' & ' fname2];
			end
		end
	end

	% (optional) Child-window data:
	if ~strcmpi(child{1},'off')
		pclist=sprintf('%s, ',child{:});
		pcdata=[]; iccount=0;
		for i=1:length(child)
			if ismember(pctype(i),[1 2])  %-- EOG or Channel data
				cdata=D.data(pc(i),:);
			elseif pctype(i)==3   %---------- IC activation
				iccount=iccount+1;
				cdata=icmult(iccount)*acts(iccount,:);
			end
			pcdata=cat(1,pcdata,cdata);
			clear cdata
		end % for i = child
		cdispch=size(pcdata,1);

		% Normalise if more than one data type:
		pctypes=unique(pctype);
		if length(pctypes)>1
			ctitle=['Continuous data: ' pclist ' magnitude normalised by data type'];
			cspc=2;
			for i=1:length(pctypes)
				t=find(pctype==pctypes(i));
				pcdata(t,:)=pcdata(t,:)/max(max(abs(pcdata(t,:))));
			end
		else
			ctitle=['Continuous data: ' pclist ' real magnitude'];
			if pctypes==1      % EOG
				cspc=500;
			elseif pctypes==2  % CH
				if isempty(altdata) % ie., main window is MEG data too
					cspc=spc;
				else
					cspc=size(pcdata,1)*max(max(abs(pcdata)));
				end
			elseif pctypes==3  % IC
				cspc=5;
			end
		end

	end % ~ child 'off'

	% NO rejections for continuous data:
	wrejcmd='';

end % if ndims
	

%%% Plot data:

% Child window:
if ~strcmpi(child{1},'off')
	cparamcmd='''srate'',srate,''dispchans'',cdispch,''winlength'',wlen,''spacing'',cspc,';
	ctitlecmd='''tag'',''childEEG'',''title'',ctitle';
	cmd1=['eegplot(pcdata,' cparamcmd ctitlecmd ');'];

	eval(cmd1);

	% Formatting:
	childfig=gcf;
	obj = findobj(childfig,'style','pushbutton'); set(obj,'Visible','off');
	p5 = findobj(childfig,'tag','Pushbutton5'); set(p5,'Visible','on');
	  p=get(p5,'Position'); p(4)=2*p(4); p(2)=p(2)+.025; set(p5,'Position',p);
	p6 = findobj(childfig,'tag','Pushbutton6'); set(p6,'Visible','on');
	  p=get(p6,'Position'); p(4)=2*p(4); set(p6,'Position',p);
	obj = findobj(childfig,'style','edit'); set(obj,'Visible','off');
	es = findobj(childfig,'tag','ESpacing'); set(es,'Visible','on');
	  p=get(es,'Position'); p(4)=3*p(4); set(es,'Position',p);
	obj = findobj(childfig,'style','text'); set(obj,'Visible','off');
	obj = findobj(childfig,'type','text'); set(obj,'FontSize',8);
	obj = findobj(childfig,'type','axes'); set(obj,'FontSize',8);
	set(childfig,'Units','normalized');set(childfig,'Position',[.025 .7 .95 .2]);
	obj=findobj(childfig,'Tag','eegaxis'); 
	  set(obj,'YTickLabel',char([' ';child(end:-1:1)']));

end

% Main window:
paramcmd='''srate'',srate,''dispchans'',dispch,''winlength'',wlen,''spacing'',spc,';
titlecmd='''title'',ptitle';
cmd=['eegplot(data,' paramcmd wrejcmd d2cmd childcmd titlecmd ');'];

eval(cmd);

% Formatting:
mainfig=gcf;
obj = findobj(mainfig,'String','REJECT'); set(obj,'Visible','off');
obj = findobj(mainfig,'style','text'); set(obj,'FontSize',8);

if ~strcmpi(child{1},'off')
	obj = findobj(mainfig,'Tag','EPosition'); set(obj,'Enable','off');
	set(mainfig,'Units','normalized');set(mainfig,'Position',[.025 .04 .95 .60]);
else
	set(mainfig,'Units','normalized');set(mainfig,'Position',[.025 .04 .95 .90]);
end

obj = findobj(mainfig,'Tag','eegaxis'); set(obj,'FontSize',8);
set(obj,'YTickLabel',chnames);


fig=mainfig;
try chfig=childfig; catch chfig=[]; end
set(0,'units','pixels')

return

