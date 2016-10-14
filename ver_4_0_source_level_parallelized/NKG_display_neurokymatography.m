%NKG_display_neurokymatography;


unixCommand = ['mne_make_movie '];
unixCommand = [unixCommand '--stcin /imaging/at03/NKG_Code_output/Version4_source_level_CUDA/VerbphraseMEG/5-neurokymography-output/averagemesh-vert2562-smooth5-elisinvsol/test-rh.stc '];
unixCommand = [unixCommand '--subject average ' ];   %average
unixCommand = [unixCommand '--jpg /imaging/at03/NKG_Code_output/Version4_source_level_CUDA/VerbphraseMEG/6-makemovie-output/test ' ];
unixCommand = [unixCommand '--view med  ' ]; %med %lat
unixCommand = [unixCommand '--tstep 5  ' ];
unixCommand = [unixCommand '--tmin -200  ' ];
unixCommand = [unixCommand '--rate 25  ' ];
unixCommand = [unixCommand '--surface inflated  ' ]; %white %inflated %pial
unixCommand = [unixCommand '--pickrange ' ];
unixCommand = [unixCommand '--smooth 5 ' ];
unixCommand = [unixCommand '--fthresh 0 '];
unixCommand = [unixCommand '--fmid 0.9 '];
unixCommand = [unixCommand '--fmax 1  --spm'];
fprintf(['[unix:] ' unixCommand '\n']);
unix(unixCommand);


