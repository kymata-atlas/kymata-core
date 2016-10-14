%NKG_display_neurokymatography;


unixCommand = ['mne_make_movie '];
unixCommand = [unixCommand '--stcin ' rootCodeOutputPath version '/' experimentName, '/5-neurokymography-output/averagemesh-vert2562-smooth5-elisinvsol/test-rh.stc '];
unixCommand = [unixCommand '--subject average ' ];   %average
unixCommand = [unixCommand '--jpg ' rootCodeOutputPath version '/' experimentName, '/6-makemovie-output/test ' ];
unixCommand = [unixCommand '--view lat  ' ]; %med %lat
unixCommand = [unixCommand '--tstep 5  ' ];
unixCommand = [unixCommand '--tmin 0  ' ];
unixCommand = [unixCommand '--rate 25  ' ];
unixCommand = [unixCommand '--surface inflated  ' ]; %white %inflated %pial
unixCommand = [unixCommand '--pickrange ' ];
unixCommand = [unixCommand '--smooth 5 ' ];
unixCommand = [unixCommand '--fthresh 0.4 '];
unixCommand = [unixCommand '--fmid 0.94 '];
unixCommand = [unixCommand '--fmax 1  --spm'];
fprintf(['[unix:] ' unixCommand '\n']);
unix(unixCommand);


