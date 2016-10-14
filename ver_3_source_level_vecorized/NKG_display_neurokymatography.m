%NKG_display_neurokymatography;


unixCommand = ['mne_make_movie '];
unixCommand = [unixCommand '--stcin /imaging/at03/NKG_Code_output/Version3_source_level_vecorized/4-neurokymatogaphy-output/elisabeth_invsol/test-lh.stc '];
unixCommand = [unixCommand '--subject average ' ];
unixCommand = [unixCommand '--jpg /imaging/at03/NKG_Code_output/Version3_source_level_vecorized/5-makemovie-output/test ' ];
unixCommand = [unixCommand '--view lat  ' ];
unixCommand = [unixCommand '--tstep 5  ' ];
unixCommand = [unixCommand '--tmin 0  ' ];
unixCommand = [unixCommand '--rate 25  ' ];
unixCommand = [unixCommand '--surface inflated  ' ]; %white %inflated %pial
unixCommand = [unixCommand '--pickrange  ' ];
unixCommand = [unixCommand '--smooth 10 ' ];
unixCommand = [unixCommand '--fthresh 0 '];
unixCommand = [unixCommand '--fmid 0.9 '];
unixCommand = [unixCommand '--fmax 1  --spm'];
fprintf(['[unix:] ' unixCommand '\n']);
unix(unixCommand);


