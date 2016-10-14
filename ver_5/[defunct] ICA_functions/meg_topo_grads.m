function [ax1 ax2 c]=meg_topo_grads(td,D,chtype,fsp);

% Plots MEG topography -- MAGS and/or GRADS -- in its own figure 
%  or in specified subplot. If chtype='both', GRADS plotted as contour lines.
%
%  INPUT:
%    td      = data vector to plot
%    D       = data structure
%    chtype  = channels to plot ('mags','grads','both')
%               default = 'both'
%               to plot chan locations only, 'chans' and td = chan indices!
%    fsp     = [fig nrows ncols subplotindex]
%               e.g., if Figure 1, subplot(4,1,1):
%               fsp = [1 4 1 1];
%               default = (new figure)
%
%  OUTPUT:
%    ax1,ax2 = axes handles for MAGS, GRADS
%    c       = contour handle for GRADS
%
%  by Jason Taylor (14/4/2008) - adapted from Rik Henson's rik_topo
%

% Options (development):
nlines=5;         % Number of contour lines (comment out for auto)
linewdth=.5;      % Width of contour lines
linecol=[1 1 1];  % Colour of contour lines (try white [1 1 1] or black [0 0 0])
doformat=1;       % Fancy format?: Contour line visibility proportional to amplitude
dotext=0;         % Add text legends (currently only if 'both')

if nargin<4
	fig=figure; hold on;
	if nargin<3
		chtype='both';
	end
else
	fig=figure(fsp(1));
	subplot(fsp(2),fsp(3),fsp(4));
end

%%% Channel stuff:

% Create channel matrix 'ch':
%  locations(1:102) X sensor type (1,2,3)
name=D.channels.name;
ch=[];
for i=1:102;
	ch=[ch;find(strncmpi(name,name{i}(1:6),6))'];
end

% Separate data by channel type:
switch chtype
	case 'mags'
		d1=td(ch(:,1));
	case 'grads'
		d2=td(ch(:,1));
		d3=td(ch(:,2));
		% Get grad vector length:
		dg=sqrt(d2.^2 + d3.^2);
	case 'both'
		d1=td(ch(:,1));
		d2=td(ch(:,2));
		d3=td(ch(:,3));
		% Get grad vector length:
		dg=sqrt(d2.^2 + d3.^2);
end

% Channel info:
load(D.channels.ctf);
Cpos = Cpos(:, D.channels.order(1:102));
x = min(Cpos(1,:)):0.005:max(Cpos(1,:));
y = min(Cpos(2,:)):0.005:max(Cpos(2,:));
[x1,y1] = meshgrid(x,y);
xp = Cpos(1,:)';
yp = Cpos(2,:)';


%%%% PLOT:

% CHANNELS:
if strcmpi(chtype,'chans')
	for i=1:length(td)
		[td(i) tmp]=find(ch==td(i));
	end
	plot3(xp(td), yp(td), td*1000, 'o',...
	 'color',[1 1 1],'markerfacecolor',[1 1 1]);
	ax1=gca;
	set(ax1,'XLim',[0 1],'YLim',[0 1]);
	axis equal off
	return
end

% MAGNETOMETERS:
if strcmpi(chtype,'mags') | strcmpi(chtype,'both')
	z = griddata(xp, yp, d1, x1, y1);
	surface(x,y,z);
	shading('interp');
	ax1=gca;
	set(ax1,'XLim',[0 1],'YLim',[0 1]);
	axis equal off
end

% GRADIOMETERS:
if strcmpi(chtype,'grads') | strcmpi(chtype,'both')
	z2 = griddata(xp, yp, dg, x1, y1);
	if strcmpi(chtype,'both')
		% Set up gradiometer contour axes:
		ax2=axes('Position',get(ax1,'Position'),...
	           'XAxisLocation','top',...
		  			 'YAxisLocation','right',...
			  		 'Color','none');
		% Draw contour plot:
		try
			nl=nlines;
			[cdata c]=contour(x,y,z2,nl,'linewidth',linewdth,'linecolor',linecol);
		catch
			[cdata c]=contour(x,y,z2,'linewidth',linewdth,'linecolor',linecol);
		end
		set(ax2,'XLim',[0 1],'YLim',[0 1]);
		axis equal off
	elseif strcmpi(chtype,'grads')
		% Draw color plot:
		surface(x,y,z2);
		shading('interp');
		ax1=gca;
		set(ax1,'XLim',[0 1],'YLim',[0 1]);
		axis equal off
	end
end

% Contour plot formatting:
if strcmpi(chtype,'both') & doformat
	% Linewidth/opaqueness proportional to grad amplitude:
	cc=get(c,'Children');
	ll=get(c,'LevelList');
	lstep=get(c,'TextStep');
	for i=1:length(ll)
%		set(findobj(fig,'userdata',ll(i)),'linewidth',(i/length(ll))*4);
		set(findobj(fig,'userdata',ll(i)),'edgealpha',i/length(ll));
	end
end
if strcmpi(chtype,'both') & dotext
  % Legend label scales:
	mtxt=sprintf('MAG Colour Scale %g : %g',round(get(ax1,'clim')));
	gtxt=sprintf('GRAD Contour Scale %g : %g : %g',round([ll(1),lstep,ll(end)]));
	text(0.4,-.1,mtxt,'HorizontalAlign','right');
	text(0.6,-.1,gtxt,'HorizontalAlign','left');
end


