%NKG_display_neurokymatography;


unixCommand = ['mne_make_movie '];
unixCommand = [unixCommand '--stcin ' rootCodeOutputPath version '/' experimentName, '/6-neurokymography-output/test-rh.stc '];
unixCommand = [unixCommand '--subject average ' ];   %average
unixCommand = [unixCommand '--jpg ' rootCodeOutputPath version '/' experimentName, '/7-plot-and-movie-output/test ' ];
unixCommand = [unixCommand '--view med  ' ]; %med %lat
unixCommand = [unixCommand '--tstep 5  ' ];
unixCommand = [unixCommand '--tmin 100  ' ];
unixCommand = [unixCommand '--rate 25  ' ];
unixCommand = [unixCommand '--surface inflated  ' ]; %white %inflated %pial
unixCommand = [unixCommand '--pickrange ' ];
unixCommand = [unixCommand '--smooth 5 ' ];
unixCommand = [unixCommand '--fthresh 0.5 ']; %0.9 (loudness)       0.995 (F0)      0.7 (periodicity)
unixCommand = [unixCommand '--fmid 0.9 ']; %0.9995 (loudness)    0.99999995 (F0) 0.999 (periodicity)
unixCommand = [unixCommand '--fmax 1  --spm'];
fprintf(['[unix:] ' unixCommand '\n']);
unix(unixCommand);


