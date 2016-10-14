%
% Projects selected ICA component(s) to sensor space using EEGLAB's 'icaproj'.
%   Specify either ICs to project using ic2proj or ICs to remove using ic2rem.
%   If both are defined, projected ICs are setdiff(ic2proj,ic2rem). NOTE: ICA 
%   must have already been run (spm_eeglab_runica). For identification of
%   blink-related IC, see spm_eeglab_icablink. See 'help icaproj' for more.
%
% FORMAT:
%   [proj D] = spm_eeglab_icaproj(S);
%
% INPUT: S (optional) structure with subfields {defaults}:
%   D         - ['string'] data filename (with 'ica' subfield) {prompt}
%   ic2proj   - [1:Nica] component(s) to be projected {0 (all)}
%   ic2rem    - [1:Nica] component(s) to remove {[] (none)}
%   samppct   - [0-1] proportion of data to use {1}, OR:
%   swin      - [1 Nsamples] sample window to use {0 (all)}, OR:
%   twin      - [1 Nsamples*(1000/D.Radc)] time window (ms) {0 (all)}
%   epochs    - [1:Nevents] epochs to use (3D data only) {0 (all)}
%   newfname  - ['string'] filename to write projected data {[] (none)}
%
% OUTPUT: [proj D]; data also saved in SPM structure if S.newfname given
%   proj      - matrix (chans x time [x epochs])
%   D         - SPM structure with projected data (if newfname given)
%   
% EXAMPLES:
%  - Project a single component:
%     S.D = 'my_raw_data.mat';
%     S.ic2proj = 2;
%     proj = spm_eeglab_icaproj(S);
%
% - Remove a single component, save as new file:
%     S.D = 'my_raw_data.mat';
%     S.ic2rem = 2;
%     S.newfname = 'remic2_my_raw_data.mat';
%     [proj D] = spm_eeglab_icaproj(S);
%
% By Jason Taylor (4/2/2008)
%  JT updated (3/3/2008) - renamed spm_eeglab_icaproj
%                        - added newfname, write utility
%   LB added ecg channel (15/7/2009)

function [proj D] = spm_eeglab_icaproj_eeg(S);


%-------------------------------------------------------
% Get Parameters:
%---------------

try    fname=S.D;
catch  fname=spm_select(1,'any','Select SPM data file for IC component projection:');
end
D=spm_eeg_ldata(fname);
[p fstem ext] = fileparts(fullfile(D.path,filesep,D.fname));
srate=D.Radc;
ndims=length(size(D.data));
nsamp=D.Nsamples;
nevents=D.Nevents;
nica=size(D.ica.W,1);

try    ecg=D.channels.ecg;
catch  ecg=[];
end

try    ic2proj=S.ic2proj;
catch  ic2proj=0;
end

try    ic2rem=S.ic2rem;
catch  ic2rem=[];
end

try    newfname=S.newfname;
catch  newfname=[];
end

w=0;
try    swin=S.swin;
       w=w+any(swin>0);
catch  swin=0;
end

try    twin=S.twin;
       w=w+any(twin>0);
catch  twin=0;
end

try    samppct=S.samppct;
       w=w+any(samppct>0);
catch  samppct=1;
end

if w>1
	error('Specify only ONE of S.twin | S.swin | S.sampct !')
end

try    epochs=S.epochs;
catch  epochs=[1:nevents];
end

% Open EEGLAB (to load toolboxes, etc.)
blah=which('eegplot');
if isempty(blah)
	eeglab(); 
	fig=gcf; close(fig);
end


%-------------------------------------------------------
% Project IC using icaproj:
%-------------------------

%%% Get data:

% Select IC(s):
if ic2proj==0
	ic2proj=[1:nica];
end
ic2proj=setdiff(ic2proj,ic2rem);

% Select Time-window:
if any(twin~=0)
	% Time-window specified:
	swin=twin/(1000/srate);
elseif samppct<1
	% Sample proportion specified:
	swin=[1 ceil(samppct*nsamp)];
elseif swin==0
	% None specified:
	swin=[1 nsamp];
end

% Get data:
if ndims==3
	data=D.data(:,swin(1):swin(end),epochs);
	data=reshape(data,size(data,1),size(data,2)*size(data,3));
elseif ndims==2
	data=D.data(:,swin(1):swin(end));
end

icachans=D.ica.chans;
weights=D.ica.W*D.ica.sphere;


%%% Project selected ICs:

proj = icaproj(data(icachans,:),weights,ic2proj);


%-------------------------------------------------------
% Write data to new SPM structure:
%--------------------------------

if ~isempty(newfname)

	[p,fstem,ext]=fileparts(newfname);
	D.fname=newfname;
	D.fnamedat=[fstem '.dat'];

	eog=[D.channels.veog D.channels.heog];

%	if icachans~=D.channels.eeg   
		D.channels.eeg=icachans;
		D.Nchannels=length(icachans)+length(eog)+length(ecg);
		D.channels.name=D.channels.name([icachans eog ecg],:);
% 		D.channels.Weight=D.channels.weight([icachans eog ecg] ,:);
% 		D.channels.scaled=D.channels.scaled([icachans eog ecg],:);
% 		D.channels.Loc=D.channels.Loc(:,icachans);
% 		D.channels.Orient=D.channels.Orient(:,icachans);
% 		D.channels.pos3D=D.channels.pos3D(icachans,:);
% 		D.channels.ort3D=D.channels.ort3D(icachans,:);
%		D.channels.order=???
%	end
	if ndims==3
		D.Nevents=length(epochs);
		try D.events.time=D.events.time(epochs); end % fails if average/contrast
		D.events.code=D.events.code(epochs);
		D.events.types=unique(D.events.code);
		D.events.Ntypes=length(D.events.types);
		D.Nsamples=nsamp;
		proj=reshape(proj,length(icachans),nsamp,nevents);
		eogdat=reshape(data(eog,:),length(eog),nsamp,nevents);
        ecgdat=reshape(data(ecg,:),length(ecg),nsamp,nevents);
	else
		D.Nsamples=size(proj,2);
		eogdat=data(eog,:);
        ecgdat=data(ecg,:);
	end
	
	D.scale = ones(D.Nchannels, 1, D.Nevents);
	D.datatype  = 'float32';

	% Add EOG & ECG:
	nrows=size(proj,1);
	proj=[proj;eogdat;ecgdat];

	% Write data to .dat file:
	fpd = fopen(fullfile(D.path, D.fnamedat), 'w');

	if ndims==3
		% Write epoched/averaged data:
		for e = 1:D.Nevents
	  	for s = 1:D.Nsamples
	    	fwrite(fpd, proj(:,s,e), 'float');
			end
		end
	else
		% Write continuous data:
		for s = 1:size(proj,2)
			fwrite(fpd, proj(:,s), 'float');
		end
	end
	fclose(fpd);

	% Store index of projected/removed component(s):
	D.ica.projected = ic2proj;
	D.ica.removed = ic2rem;

	if str2num(version('-release'))>=14 
		save(fullfile(D.path, D.fname), '-V6', 'D');
	else
		save(fullfile(D.path, D.fname), 'D');
	end

	% Remove EOG/ECG:
	if ndims==3
		proj=proj(1:nrows,:,:);
	else
		proj=proj(1:nrows,:);
	end
	
end


%-------------------------------------------------------
% Return proj matrix:
%-------------------

% Reshape if epoched:
if ndims==3
	proj=reshape(proj,length(icachans),size(data,2),size(data,3));
end



%
%%-------------------------------------------------------
%% Plot continuous projected data:
%%-------------------------------
%
%if plotchans>0
%
%	P.D=fname;
%	P.data=proj;
%	P.args=plotargs;
%	
%	spm_eeglab_eegplot(P);
%
%% OLD:
%%	ttl=['Continuous Plot: Projection of IC(s) ', num2str(ic2proj)];
%%	eegplot(proj(plotchans,:),'srate',D.Radc,'dispchans',20,...
%%	  'winlength',10,'spacing',1000,'title',ttl);
%
%end
%
%
%%-------------------------------------------------------
%% Plot topography:
%%----------------
%% (stolen from rik_topo, which is a modification of spm_eeg_scalp2d)
%
%%%% UPDATE THIS WITH spm_eeglab_plottopo!!
%
%if topochans>0
%
%	% Get sensor locations:
%	load(D.channels.ctf);
%	Cpos = Cpos(:, D.channels.order(topochans));
%	x = min(Cpos(1,:)):0.005:max(Cpos(1,:));
%	y = min(Cpos(2,:)):0.005:max(Cpos(2,:));
%
%	% Create 2D grid:
%	[x1,y1] = meshgrid(x,y);
%	xp = Cpos(1,:)';
%	yp = Cpos(2,:)';
%
%	switch oper
%	case {'mean'}
%		td=mean(proj(topochans,:),2);
%		oper='MEAN';
%	case {'var'}
%		td=var(proj(topochans,:),0,2);
%		oper='VARIANCE';
%	end
%		
%	z = griddata(xp, yp, td, x1, y1);
%
%	% Plot it!
%	figure
%	surface(x,y,z);
%	shading('interp')
%	hold on
%	plot3(xp, yp, td, 'k.');
%	ch=colorbar;
%	axis off
%	
%	title(['Topography: ' oper ' of Projection of IC(s) ', num2str(ic2proj)]);
%
%end
%
%

