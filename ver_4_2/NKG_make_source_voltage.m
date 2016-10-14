

addpath /imaging/local/software/mne/mne_2.7.3/x86_64/MNE-2.7.3-3268-Linux-x86_64/share/matlab/;
addpath /imaging/at03/NKG_Code/Version4_2/mne_matlab_functions/;

%% Script to find energies of grecian bands of all

indir = '/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG/4-averaged-by-trial-data/vert2562-smooth5-nodepth-eliFM-snr1-signed/';
outdir = '/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG/4-averaged-by-trial-data/vert2562-smooth5-nodepth-eliFM-snr1-signed-intergral/';

%create full wordlist (i.e. all words)

wordlistFilename  = '/imaging/at03/NKG_Data_Sets/VerbphraseMEG/scripts/Stimuli-Verbphrase-MEG-Single-col.txt';
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
wordlist = wordlist{1};
fclose('all');



for w = 1:numel(wordlist)
            
            %left hand
            thiswordstc = mne_read_stc_file(strjoin([indir, wordlist(w), '-lh.stc'], ''));
            thiswordstc.data = cumtrapz(thiswordstc.data, 2);
            mne_write_stc_file(strjoin([outdir, wordlist(w), '-lh.stc'], ''), thiswordstc);

            clear envelopes thiswordstc;    
            
            %right hand    
            thiswordstc = mne_read_stc_file(strjoin([indir, wordlist(w), '-rh.stc'], ''));
            thiswordstc.data = cumtrapz(thiswordstc.data, 2);
            mne_write_stc_file(strjoin([outdir, wordlist(w), '-rh.stc'], ''), thiswordstc);
            
            clear envelopes thiswordstc; 
end