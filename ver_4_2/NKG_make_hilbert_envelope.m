

addpath /imaging/local/software/mne/mne_2.7.3/x86_64/MNE-2.7.3-3268-Linux-x86_64/share/matlab/;
addpath /imaging/at03/NKG_Code/Version4_2/mne_matlab_functions/;

%% Script to find energies of grecian bands of all

indir = '/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG/4-averaged-by-trial-data/frequency_band_encodings/vert2562-smooth5-nodepth-eliFM-snr1-signed/';
outdir = '/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG/4-averaged-by-trial-data/frequency_band_encodings/vert2562-smooth5-nodepth-eliFM-snr1-signed-hilbert-envelope/';
bands = cellstr([

    'alphaWave';
    'betaWave ';
    'deltaWave';
    'gammaWave';
    'thetaWave';
    
]);


%create full wordlist (i.e. all words)

wordlistFilename  = '/imaging/at03/NKG_Data_Sets/VerbphraseMEG/scripts/Stimuli-Verbphrase-MEG-Single-col.txt';
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
wordlist = wordlist{1};
fclose('all');



for b = 1:numel(bands)
    for w = 1:numel(wordlist)
            
            %left hand
            thiswordstc = mne_read_stc_file(strjoin([indir, deblank(bands(b)), '/' , wordlist(w), '-lh.stc'], ''));
            envelopes = abs(hilbert(thiswordstc.data'));
            thiswordstc.data = envelopes';
            mne_write_stc_file(strjoin([outdir, deblank(bands(b)), '/' , wordlist(w), '-lh.stc'], ''), thiswordstc);

            clear envelopes thiswordstc;    
            
            %right hand    
            thiswordstc = mne_read_stc_file(strjoin([indir, deblank(bands(b)), '/' , wordlist(w), '-rh.stc'], ''));
            envelopes = abs(hilbert(thiswordstc.data'));
            thiswordstc.data = envelopes';
            mne_write_stc_file(strjoin([outdir, deblank(bands(b)), '/' , wordlist(w), '-rh.stc'], ''), thiswordstc);
            
            clear envelopes thiswordstc; 
    end
end