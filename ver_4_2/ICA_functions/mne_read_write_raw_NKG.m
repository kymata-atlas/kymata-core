function mne_read_write_raw_NKG(RawFile,OutFile,new_data, tempdigi)
%
% function mne_ex_read_write_raw(infile,outfile);
%
% Read and write raw data in 60-sec blocks
%
%   Copyright 2007
%
%   Matti Hamalainen
%   Athinoula A. Martinos Center for Biomedical Imaging
%   Massachusetts General Hospital
%   Charlestown, MA, USA
%
%   No part of this program may be photocopied, reproduced,
%   or translated to another program language without the
%   prior written consent of the author.
%
%   $Id: mne_ex_read_write_raw.m 2628 2009-04-27 21:17:31Z msh $
%
% EF 04/11 Adapt for ICA recombination of data
%

global FIFF;
if isempty(FIFF)
   FIFF = fiff_define_constants();
end
%
me = 'MNE:mne_ex_read_write_raw';
%
% if nargin ~= 2
%     error(me,'Incorrect number of arguments');
% end
%
%   Setup for reading the raw data
%
try
    raw = fiff_setup_read_raw(RawFile);
catch
     error(me,'%s',mne_omit_first_line(lasterr));
end


% AT's's change - re-insert digitiser data!
tempdigidata = fiff_read_evoked(tempdigi);
raw.info.dig = tempdigidata.info.dig;  



%raw.info.projs = [];
%
%   Set up pick list: MEG + EEG + STI 0101 + EOG - bad channels
%
%
    want_meg   = true;
    want_eeg   = true;
    want_stim  = false;

    include{1} = 'STI101';
    include{2} = 'EOG061';
    include{3} = 'EOG062';
    try
        picks = fiff_pick_types(raw.info,want_meg,want_eeg,want_stim,include);
    catch
        error(me,'%s (channel list may need modification)',mne_omit_first_line(lasterr));
    end
    
%
[outfid,cals] = fiff_start_writing_raw(OutFile,raw.info,picks);
%
%   Set up the reading parameters
%
from        = raw.first_samp;
to          = raw.last_samp;
quantum_sec = 10;
quantum     = ceil(quantum_sec*raw.info.sfreq);
%
%   To read the whole file at once set
%
%quantum     =ceil(to - from + 1); % does not work properly
%
%
%   Read and write all the data
%
time_end = size(new_data,2);
t=1;
first_buffer = true;
for first = from:quantum:to
    last = first+quantum-1;
    
    first_new =t*quantum-quantum+1;
    last_new = first_new+quantum-1;
    if last > to
        last = to;
        last_new = time_end;
    end
    try
        [ data, times ] = fiff_read_raw_segment(raw,first,last,picks);
        
        data_temp = new_data(:,first_new:last_new);
        t=t+1;
        
% TRIGGER addition
                    
        trig = data(377,:);
        data_temp(377,:) = trig;
        data = data_temp;
        clear trig  data_temp
        
    catch
        fclose(raw.fid);
        fclose(outfid);
          error(me,'%s',mne_omit_first_line(lasterr));
    end
    %
    %
    fprintf(1,'Writing...');
    if first_buffer
       if first > 0
	  fiff_write_int(outfid,FIFF.FIFF_FIRST_SAMPLE,first);
       end
       first_buffer = false;
    end
    fiff_write_raw_buffer(outfid,data,cals);
    fprintf(1,'[done]\n');
end
clear t
fiff_finish_writing_raw(outfid);
fclose(raw.fid);
