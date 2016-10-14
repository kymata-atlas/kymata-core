%
% Classify and/or remove artefact-related independent component(s). 
%   Correlates each IC's activation timecourse with physiological (E*G) data. 
%   Component is recommended as e.g. blink component if corr w/VEOG
%   (raw and Z-score) exceeds a user-set threshold. Optionally plots 
%   IC and E*G timecourses for inspection. NOTE: ICA must have already
%   been run (spm_eeglab_runica). 
%
% INPUT: S (optional) structure with (optional) subfields [<default>]:
%   D             - data filename (with 'ica' subfield) [<prompt>]
%   mode          - {'classify'|'remove'|'both'|'view'} [<{'classify'}>]
%                 -  view is same as classify but doesn't save!
%   plotraw       - plot raw data first? (1|0) [<0>]
%   plotchans     - channels to plot 
%                 -  mags: [<52 29 1 42 15 101 80 57>]
%                 -  grds: [<51 52 29 30 1 2 41 42 15 16 101 102 79 80 57 58>]
%                 -  eeg:  [<8 2 4 36 32 61 70 51>]
%                    (used for raw and diff plots)
%		cpos          - 'position' of continuous data fig [x y w h] (normalized)
%   chpos         - 'position' of child (E*G) fig
%   tpos          - 'position' of hist/topo fig
%
% -- mode: 'classify'|'both' -------------------------------------
%
% INPUT: S subfields: 
%   artlabs       - Labels of artefacts to classify, e.g.:
%                    {'blink' 'horiz' 'pulse' 'other'}  [<prompt>]
%   chanlabs      - Names of physiological signal channels (cellstr), e.g:
%                    {'veog' 'heog' 'ecg' 'other'}  [<prompt>]
%                    expects D.channels.<label> to exist
%                 OR vector of channel indices, e.g.:
%                    [308 307 ...]
%   rthresh       - vector (or scalar) of corr thresholds [<.5>]
%   zthresh       - vector (or scalar) of Z-score thresholds [<3>]
%                    NOTE: thresholds merely determine recommendations
%   excrej        - exclude rejected trials from corrs? (1|0) [<0>]
%                    NOTE: applies to epoched data only
%   plotacts      - plot ICA/VEOG activations? (1|0) [<1>]
%   plothist      - plot histogram of corr Z-scores (1|0) [<1>]
%   plottopo      - plot topographies of high-r components? (1|0) [<1>]
%
% OUTPUT:
%   D (and saved .mat) with added/modified:
%    ica.class.<artlab>  - index of artefact IC, eg., ica.class.blink
%    ica.class.r<artlab> - corr of IC(s) & E*G, eg.,  ica.class.rblink
%    ica.class.z<artlab> - Z-score IC-E*G corr, eg.,  ica.class.zblink
%   afig - handles to IC activation figure & E*G figure
%
%   -- mode: 'remove'|'both -----------------------------------------
%
% INPUT: S subfields:  
%   remlabs       - labels of artefact ICs to remove, e.g. {'blink'} [<prompt>]
%                   should exist (or be created) as fields in D.ica.class
%   plotdiff      - plot clean & dirty data overlayed? (1|0) [<1>]
%   remfname      - name of new 'clean' filename [<'rem_' S.D>]
%
% OUTPUT:
%   D (and saved .mat) with added/modified:
%    ica.class.removed  - index of artefact IC(s) removed
%    ica.class.remlabs  - label of artefact IC(s) removed
%
% Created by Jason Taylor (02/03/2009 ... from meg_icablink)
% ... see /imaging/jt03/public/demo for more info
%

function [D] = meg_ica_artefact_grds(S);

% Hidden option (USE IS NOT RECOMMENDED!)
% S.noninteractive  - accept all recommendations, close all plots (* in development)
%


%-- Initialise -----------------------------------------------------------

% Check for SPM:
try   Finter=findobj(0,'Tag','Interactive');
catch spm eeg;
			pause(5)
		  Finter=findobj(0,'Tag','Interactive');
end

% Open EEGLAB (to load toolboxes, etc.)
tmp=which('eegplot'); 
if isempty(tmp)
	eeglab();
	close(findobj(0,'Tag','EEGLAB')); % close EEGLAB window
end

% Data file:
try fn=S.D;
catch fn=spm_select(1,'^*ica.*mat','Select data file for ICA artefact ID:');
end
D=spm_eeg_ldata(fn);
srate=D.Radc;
ndims=length(size(D.data));
chans=D.ica.chans;
[p fstem ext] = fileparts(fullfile(D.path,filesep,D.fname));
% Get channel type:
if strcmpi(fstem(end-3:end),'mags')
	chtype='mags';
	spc=5000; % spacing for figures
elseif strcmpi(fstem(end-3:end),'grds')
	chtype='grads';
	spc=3000;
elseif strcmpi(fstem(end-2:end),'eeg')
	chtype='eeg';
	spc=100;  
elseif strcmpi(fstem(end-3:end),'grms')
	chtype='grms';
	error('This function is not meant to be run on grad RMS data!')
else
	chtype='both';
	spc=5000;	
end
cd(p)

% Mode
try   domode=S.mode;
catch domode={'classify'};
end
domode=cellstr(char(domode));

% Plot raw data first?
try   plotraw=S.plotraw;
catch plotraw=0; %%%%%%% changed to 0 %%%  
end

% Channels to plot (for raw and/or diff)
try   plotchans=S.plotchans;
catch if strcmpi(chtype,'mags') | strcmpi(chtype,'both')
				plotchans=[52 29 1 42 15 101 80 57]; 
			elseif strcmpi(chtype,'grads')
				plotchans=[51 52 29 30 1 2 41 42 15 16 101 102 79 80 57 58];
			elseif strcmpi(chtype,'eeg')
				plotchans=[8 2 4 36 32 61 70 51];
			end
end

% Hack for nice figure position:
try   cpos=S.cpos;
catch cpos=[0.322 0.0426 0.66 0.4691];
end

try   chpos=S.chpos;
catch chpos=[0.322 0.5723 0.66 0.2];
end

try   tpos=S.tpos;
catch tpos=[0.2 0.45 0.52 0.45];
end


%--- View raw data -------------------------------------------------------

if plotraw

	spm_input(sprintf('Viewing raw data: %s%s',fstem,ext),1,'d');
	drawnow

	clear P
	P.D=fn;
	P.chans=plotchans;
	P.wlen=10;
	P.spc=spc;
	P.child={};
 	childch={'veog' 'heog'};  % CW removed ECG: childch={'veog' 'heog' 'ecg'}; 
	for i=1:length(childch)
		if isfield(D.channels,childch{i})
			P.child=[P.child childch{i}];
		end
	end

	[pfig pchfig]=meg_plotcnt_editbycw(P);

	% Adjust figure size:
	set(pfig,'position',cpos);
	try  set(pchfig,'position',chpos); end

	% Plot chans in topoplot:
	tchfig=figure;
	set(tchfig,'color',[.5 .5 .5],'units','normalized','position',[0.0675 0.4745 0.2484 0.2628]);
	meg_topo(zeros(size(D.channels.eeg)),D,chtype,[tchfig 1 1 1]);
	meg_topo(plotchans,D,'chans',[tchfig 1 1 1]);

	cont=spm_input('Inspect raw data, then...','+1','b','continue|quit',[1 2],1);
	try close(pfig);   end
	try close(pchfig); end
	try close(tchfig); end

	if cont==2
		disp(sprintf('Quitting!!'))
		return
	end
	
end % if plotraw


%--- Mode: Classify artefact ICs -------------------------------------------

if strcmpi(domode,'classify') | strcmpi(domode,'both') | strcmpi(domode,'view')

	spm_input(sprintf('ICA Artefact Classification: %s%s',fstem,ext),1,'d');
	drawnow
	
	% Get list of artefacts to classify:
	try   artlabs=S.artlabs;
	catch artlabs=spm_input('Artefact labels sep by commas:','+1','s','blink, horiz');
	end
	if ~iscellstr(artlabs)
		% deblank:
		artlabs=artlabs(artlabs~=' ');
		% turn into cell string:
		ind=[0 findstr(artlabs,',') length(artlabs)+1];
		tmp={''};
		for i=1:length(ind)-1
			tmp{i}=artlabs(ind(i)+1:ind(i+1)-1);
		end
		artlabs=tmp;
	end
	narts=length(artlabs);

	% Get E*G or other reference channels:
	try   chanlabs=S.chanlabs;
	catch chanlabs=spm_input('Channel labels sep by commas:','+1','s','veog, heog');
	end
	if ~iscellstr(chanlabs)
		% deblank:
		chanlabs=chanlabs(chanlabs~=' ');
		% turn into cell string:
		ind=[0 findstr(chanlabs,',') length(chanlabs)+1];
		tmp={''};
		for i=1:length(ind)-1
			tmp{i}=chanlabs(ind(i)+1:ind(i+1)-1);
		end
		chanlabs=tmp;
	end
	nechans=length(chanlabs);
	if nechans<narts
		for i=nechans:narts
			chanlabs{i}=[];
		end
	end
	% Check whether E*G channels are valid:
	echans={[]}; echanvalid=[];
	for i=1:nechans
		try	  eval(sprintf('echans{i}=D.channels.%s;',chanlabs{i}));
					echanvalid(i)=1;
		catch echans{i}=[];
		      warning(sprintf('\nCannot find D.channels.%s\n',chanlabs{i}));
					disp(sprintf('(no correlation will be computed)'))
					echanvalid(i)=0;
		end
	end
	echans=cell2mat(echans);

	% Threshold options:
	try rthresh=S.rthresh; 	catch rthresh=.5; end
	try zthresh=S.zthresh;  catch zthresh=3;  end

	if length(rthresh) < nechans
		disp(sprintf('Using rthresh %g for all channels',rthresh(1)))
		rthresh=repmat(rthresh(1),1,nechans);
	end
	if length(zthresh) < nechans
		disp(sprintf('Using zthresh %g for all channels',zthresh(1)))
		zthresh=repmat(zthresh(1),1,nechans);
	end
		
	% Plot options:
	try plotacts=S.plotacts;  catch plotacts=1; end   
	try plottopo=S.plottopo;  catch plottopo=1; end	%!!!changed to 1 for MEG% 
	try plothist=S.plothist;  catch plothist=1; end
	
	% Exclude rejected trials from corr?:
	try excrej=S.excrej;  catch excrej=0; end
	

	%%% Plot ICA Activations %%%
	if plotacts
		clear E
		E.D=fn;
		E.altdata={'acts'};
		E.dispch=10;
		E.spc=10;
		if any(echanvalid)
			E.child=chanlabs(find(echanvalid));
		else
			E.child={'off'};
		end
		[afig achfig acts]=meg_plotcnt_editbycw(E);
		
		% Adjust figure size:
 		set(afig,'position',cpos);
 		try	set(achfig,'position',chpos); end
		
		cont=spm_input('Inspect IC activations, then...','+1','b','continue|quit',[1 2],1);
		if cont==2
			disp(sprintf('Quitting!'))
			return
		end
	else
		% Don't plot, just get activations
		afig=[];
		clear A
		A.D=fn;
		acts=spm_eeglab_icaact(A);
	end
	
	
	% Need to fix for epoched data and for excrej! ****
	try edata=D.data(echans,:);
	catch docorr=0;
	end
	

%if ndims==3
%	acts=meg_icaact(A);
%%	data=D.data(chans,:,:);
%%	data=reshape(data,length(chans),size(data,2)*size(data,3));
%	epindex=1:D.Nevents;
%	if userej
%		epindex=epindex(~D.events.reject);
%	end
%	edata=squeeze(D.data(echans,:,epindex));
%	edata=reshape(edata,nechans,prod(size(edata))); % check this!
%%	vdata=squeeze(D.data(vchan,:,epindex));
%%	hdata=squeeze(D.data(hchan,:,epindex));
%%	vdata=reshape(vdata,1,size(vdata,1)*size(vdata,2));
%%	hdata=reshape(hdata,1,size(hdata,1)*size(hdata,2));
%elseif ndims==2
%	acts=meg_icaact(A);
%%	data=D.data(chans,:);
%	edata=D.data(echans,:);
%%	vdata=D.data(vchan,:);
%%	hdata=D.data(hchan,:);
%end

	nica = size(acts,1);
	
	
	%%% Compute correlations with E*G:
	%for c=1:nechans
	for c=1:narts
		if ~echanvalid(c)
			disp(sprintf('\nNo channel given for %s',artlabs{c}))
			disp(sprintf('..Correlations cannot be computed!'))
			docorr=0;
		else
			disp(sprintf('\nComputing ICA-%s correlations...\n',chanlabs{c}))
			docorr=1;
		end
		
		if docorr
			rvec=[];
			for i=1:nica
				r=corr(acts(i,:)',edata(c,:)');
				rvec=[rvec; r];
			end
			zvec=zscore(abs(rvec));
				
			% Store top three indices & values:
			rsort = sort(abs(rvec),'descend');
			itop=[find(abs(rvec)==rsort(1)) find(abs(rvec)==rsort(2)) find(abs(rvec)==rsort(3))];
			rtop=rsort(1:3);
			ztop=zvec(itop);
		
			%%% Plot Z-score histogram:
			if plothist
				try   figure(tfig);
				      clf(tfig);
				catch tfig=figure; set(tfig,'color',[1 1 1],'units','normalized','position',tpos);
				end
				subplot(2,1,2);
				colormap(jet);
				disp(sprintf('Plotting Zscore histogram...'))
				hist(zvec,50);
				hax=gca; set(hax,'box','off');
				xlabel(sprintf('Z-score of correlation with %s',upper(chanlabs{c}))); 
				ylabel('Frequency')
		
				% Label high outliers:
				ztxt=[];
				for i=1:3
					if zvec(itop(i))>1
						str=sprintf('IC%d',itop(i));
						ztxt(i)=text(zvec(itop(i)),4,str);
						set(ztxt(i),'horizontalalign','center','fontweight','bold');
					end
				end
				drawnow
				
			end %if plothist
		
			%%% Topographies
			if plottopo
				disp(sprintf('Plotting topographies...'))
				try   figure(tfig);
				catch tfig=figure; set(tfig,'color',[1 1 1],'units','normalized','position',tpos);
				end
				
				if ~plothist
					clf(tfig);
				end
					
				if strcmpi(chtype,'both')
					topoch={'mags','grads'};
					sprows=4;
				else
					topoch=cellstr(chtype);
					sprows=2;
				end
				
				tax=[];ttxt=[];
				for i=1:3
					disp(sprintf('Projecting IC%d to sensors',itop(i)))
					clear P
					P.D=fn;
					P.ic2proj=itop(i);
					proj=spm_eeglab_icaproj(P);
					proj=mean(proj,2);
					
					for j=1:length(topoch)
						ch=topoch{j};
						fsp=[tfig sprows 3 ((j*3)+1-i)];
						tax(i)=meg_topo_grads(proj,D,ch,fsp);   %%%%%%%%%%%%%%% CW changed to meg_topo_grads from meg_topo
						
						if j==1
							% Label:
							str=sprintf('IC%d\nr=%0.3f\nZ=%0.3f',itop(i),abs(rvec(itop(i))),zvec(itop(i)));
							try eval(sprintf('previc=D.ica.class.%s;',artlabs{c}));
									if any(previc==itop(i))
										str=sprintf('%s\n''%s''',str,artlabs{c});
									end
							end
						elseif j==2
							str='grms';
						end % if j
		
						ttxt(i)=text(.5,0,str);
						set(ttxt(i),'horizontalalign','center','verticalalign','top','fontweight','bold');
						drawnow

					end %for j

					clear proj

				end % for i

			end % if plottopo

			% Display summary table:
			disp(sprintf('\n================================'))
			disp(sprintf('Top 3 Correlations with %s',chanlabs{c}))
			disp(sprintf('--------------------------------'))
			disp(sprintf(' IC\tCORR\tZ-SCORE'))
			disp(sprintf('--------------------------------'))
			disp(sprintf(' %d\t%0.3f\t%0.3f',itop(1),rtop(1),ztop(1)))
			disp(sprintf(' %d\t%0.3f\t%0.3f',itop(2),rtop(2),ztop(2)))
			disp(sprintf(' %d\t%0.3f\t%0.3f',itop(3),rtop(3),ztop(3)))
			disp(sprintf('--------------------------------\n'))
			disp(sprintf('Mark artefact IC using spm interactive window...'))

			% Display on spm interactive window:
			str=sprintf('RESULTS FOR %s/%s',upper(artlabs{c}),upper(chanlabs{c}));
			spm_input(str,2,'d');
			str=sprintf('..r-values: IC%d %0.3f | IC%d %0.3f | IC%d %0.3f',...
				itop(1),rtop(1),itop(2),rtop(2),itop(3),rtop(3));
			spm_input(str,'+1','d');
			str=sprintf('..Z-scores: IC%d %0.3f | IC%d %0.3f | IC%d %0.3f',...
				itop(1),ztop(1),itop(2),ztop(2),itop(3),ztop(3));
			spm_input(str,'+1','d');
				
			% Recommendation:
			if rtop(1)>rthresh(c) & ztop(1)>zthresh(c)
				if rtop(2)<rthresh(c) & ztop(2)<zthresh(c)
					rec=itop(1); recstr='Suprathreshold and unique';
				elseif rtop(2)>rthresh(c) | ztop(2)>zthresh(c)
					rec=[itop(1) itop(2)]; recstr='Suprathreshold but NOT unique!';
				end
			elseif rtop(1)>rthresh(c) & ztop(1)<zthresh(c)
				if rtop(2)<rthresh(c)
					rec=itop(1); recstr='Suprathreshold (r) but SUBthreshold (Z)';
				elseif rtop(2)>rthresh(c)
					rec=[itop(1) itop(2)]; recstr='Suprathreshold (r) but SUBthreshold (Z) and NOT unique!';
				end
			elseif ztop(1)>zthresh(c) & rtop(1)<rthresh(c)
				if ztop(2)<zthresh(c)
					rec=itop(1); recstr='Suprathreshold (Z) but SUBthreshold (r)';
				elseif ztop(2)>zthresh(c)
					rec=[0]; recstr='Suprathreshold (Z) but SUBthreshold (r) and NOT unique!';
				end
			elseif rtop(1)<rthresh(c) & ztop(1)<zthresh(c)
				rec=0; recstr='SUBthreshold (r & Z)!!'
			end
			
			% Ask user which to mark:	
			str=sprintf('Mark IC(s) as %s (0=none):',artlabs{c});
			%def='0';         % no recommendations
			def=num2str(rec); % recommendations
			spm_input(recstr,'+1','d');
			umark=spm_input(str,'+1','w',def,[1 Inf],nica);
		
			if all(umark>0)
				eval(sprintf('D.ica.class.%s = [%s];',artlabs{c},num2str(umark)));
				eval(sprintf('D.ica.class.r%s= [%s];',artlabs{c},num2str(rvec(umark)')));
				eval(sprintf('D.ica.class.z%s= [%s];',artlabs{c},num2str(zvec(umark)')));
				disp(sprintf('\nMarking [%s] as %s',num2str(umark),artlabs{c}))
			else
				eval(sprintf('D.ica.class.%s = []',artlabs{c}));
				disp(sprintf('\nNot marking any component as %s',artlabs{c}))
			end
		
			% Report classifications:
			D.ica.class
		
		else % if docorr
		
			% Display on spm interactive window:
			str=sprintf('ARTEFACT ''%s'': NO CHAN ''%s'' FOR CORRELATIONS',upper(artlabs{c}),upper(chanlabs{c}));
			spm_input(str,2,'d');
			str=sprintf('..Select candidate ICs to view topogaphies:');
			spm_input(str,'+1','d');
			% User may request to see topos:
			str=sprintf('Which IC(s) for topo? (0=none):');
			tview=spm_input(str,'+1','w','0',[1 Inf],nica);
		
			% If user requested topos:
			if all(tview>0)
				disp(sprintf('Plotting topographies...'))
				try   figure(tfig);
				      clf(tfig);
				catch tfig=figure; set(tfig,'color',[1 1 1],'units','normalized','position',tpos);
				end
				colormap(jet)
				tax=[];ttxt=[];
		
				for i=1:length(tview)
					% Project IC activation to sensors:
					clear P
					P.D=fn;
					P.ic2proj=tview(i);
					disp(sprintf('Projecting IC%d to sensors',tview(i)))
					proj=spm_eeglab_icaproj(P);
					proj=mean(proj,2);
		
					% Arrange topo position:
					sp=mod(i,9);
					if mod(i,9)==0
						sp=9;
					end
					
					% Plot topo:
					fsp=[tfig 3 3 sp];
					tax(i)=meg_topo(proj,D,chtype,fsp); 
					
					% Label:
					str=sprintf('IC%d',tview(i));
					try eval(sprintf('previc=D.ica.class.%s;',artlabs{c}));
							if any(previc==tview(i))
								str=sprintf('%s\n''%s''',str,artlabs{c});
							end
					end
					ttxt(i)=text(.5,0,str);
	%				ttxt(i)=text(.5,0,sprintf('IC%d',tview(i)));
					set(ttxt(i),'horizontalalign','center','verticalalign','top','fontweight','bold');
					drawnow
			
					clear proj
		
					% if i>9, clear plot
					if mod(i,9)==0
						cont=spm_input('Clear and plot more topos','+1','b','continue|quit',[1 2],1);
						if cont==2
							disp(sprintf('Quitting!'))
							return
						end
						try clf(tfig); end
					end
				
				end % for i in tview
						
			end % if tview>0

			% Ask user which to mark:	
			str=sprintf('Mark IC(s) as %s (0=none):',artlabs{c});
			def='0'; % could make recommendation
			umark=spm_input(str,'+1','w',def,[1 Inf],nica);
		
			if all(umark>0)
				eval(sprintf('D.ica.class.%s = [%s];',artlabs{c},num2str(umark)));
				eval(sprintf('D.ica.class.r%s= [%s];',artlabs{c},num2str(rvec(umark)')));
				disp(sprintf('\nMarking [%s] as %s',num2str(umark),artlabs{c}))
			else
				eval(sprintf('D.ica.class.%s = []',artlabs{c}));
				disp(sprintf('\nNot marking any component as %s',artlabs{c}))
			end
		
			D.ica.class
			
		end % if docorr
				
	end % c in echan
		
	if ~strcmpi(domode,'view')
		disp(sprintf('Saving ICA Classifications: %s',D.fname))
		save(fullfile(D.path,filesep,D.fname),'D')
	end
	
	% Close remaining figures:
	try close(afig);   end
	try close(achfig); end
	try close(tfig);   end

end % if domode classify | both


% --- Mode: Remove Artefact ICs ---------------------------------------------------

if strcmpi(domode,'remove') | strcmpi(domode,'both')

	spm_input(sprintf('ICA Artefact removal for %s%s',fstem,ext),1,'d');
	drawnow

	% List of artefacts to classify:
	try   remlabs=S.remlabs;
	catch remlabs=spm_input('Artefacts to remove:','+1','s','blink, horiz');
	end
	if ~iscellstr(remlabs)
		% deblank:
		remlabs=remlabs(remlabs~=' ');
		% turn into cell string:
		ind=[0 findstr(remlabs,',') length(remlabs)+1];
		tmp={''};
		for i=1:length(ind)-1
			tmp{i}=remlabs(ind(i)+1:ind(i+1)-1);
		end
		remlabs=tmp;
	end
	nrem=length(remlabs);

	spm_input(sprintf('Removing: %s',sprintf('%s ',remlabs{:})),'+1','d');
	drawnow
	
	% New filename:
	try   remfname=S.remfname;
	catch remfname=['rem_' fstem ext];
	end
			
	% Plot dirty and clean data together?
	%  (uses channels from 'plotchans')
	try   plotdiff=S.plotdiff;   
	catch plotdiff=1; 
	end

	%%% Gather artefact ICs:
	nica=size(D.ica.W,1);
	remvec=[];
	for i=1:nrem
		remlab=remlabs{i};
		try   eval(sprintf('remvec=[remvec D.ica.class.%s];',remlab))
		catch str=sprintf('Cannot find D.ica.class.%s',remlab);
			    spm_input(str,'+1','d');
					str=sprintf('Enter %s IC (0=none):',remlab);
			    uresp=spm_input(str,'+1','w','0',[1 Inf],nica);
					remvec=[remvec uresp];
					if uresp==0
						remlabs{i}='';
					end
		end
	end % for i=1:nrem
	remvec=remvec(remvec>0);
	
	%%% Project out artefact ICs:
	disp(sprintf('\nProjecting out artefact ICs'))
	clear R
	R.D=fn;
	R.ic2rem=remvec;
	R.newfname=remfname;
	[proj D] = spm_eeglab_icaproj(R);
	clear proj

	spm_input(sprintf('Data saved as: %s',remfname),'+1','d');
	drawnow

	% Store record of removed components:
	D=spm_eeg_ldata(D.fname);
	D.ica.class.removed=remvec;
	D.ica.class.remlabs=remlabs;
	save(D.fname,'D')
	
	%%% View data:
	if plotdiff
		disp(sprintf('Plotting cleaned data with original data overlayed...'))
	
		clear C
		C.D=remfname;
		C.D2=fn;
		C.chans=plotchans;
		C.wlen=10;
		C.spc=spc;
		C.child={};
		try   childch=chanlabs;
		catch childch={'veog' 'heog'};  %CW removed ECG: catch childch={'veog' 'heog' 'ecg'};  
		end
		for i=1:length(childch)
			if isfield(D.channels,childch{i})
				C.child=[C.child childch{i}];
			end
		end

		[cfig cchfig]=meg_plotcnt_editbycw(C);
		
		% Adjust size:
		set(cfig,'position',cpos);
		try	set(cchfig,'position',chpos); end
		
		% Plot chans in topoplot:
% 		tchfig=figure;    %%%%%%%%%%%%%%%%% commented out for grads %%
% 		set(tchfig,'color',[.5 .5 .5],'units','normalized','position',[0.0675 0.4745 0.2484 0.2628]);
% 		meg_topo_grads(zeros(size(D.channels.eeg)),D,chtype,[tchfig 1 1 1]);
% 		meg_topo_grads(plotchans,D,'chans',[tchfig 1 1 1]);	
		
		cont=spm_input('View clean+orig data then...','+1','b','done|quit',[1 2],1);
		try close(cfig);   end
		try close(cchfig); end
		try close(tchfig); end

		if cont==2
			disp(sprintf('Quitting!'))
			return
		end
	
	end % if plotdiff
	
end % if domode remove | both 

disp(sprintf('\nDone!\n'))

if strcmpi(domode','view')
	disp(sprintf('VIEW MODE - Data not saved!'),'+1','d')
	spm_input(sprintf('VIEW MODE - Data not saved!'),'+1','d');
	drawnow
end

return	

