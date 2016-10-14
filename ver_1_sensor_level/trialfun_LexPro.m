function [trl, event] = trialfun_LexPro(cfg)

hdr   = read_header(cfg.dataset);
event = cfg.trialdef.events; % don't get from data

% determine the number of samples before and after the trigger
pretrig  = -cfg.trialdef.pre * hdr.Fs;
posttrig =  cfg.trialdef.post * hdr.Fs;

sel = find(strcmp({event.type}, 'stimuli'));

trl = [];
for i=1:length(sel)
  begsample = event(sel(i)).sample + pretrig;
  %endsample = begsample - pretrig + posttrig + event(sel(i)).duration;
  endsample = begsample - pretrig + posttrig + 900;
  offset = pretrig;
  trl = [trl; [begsample endsample offset]];
end