%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% run raw data through maxfilter 2.2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% based partly on Jason's script and partly on Elisabeth's.

for p = 1:numel(participentIDlist) 
               
    for b = 1:participentNumBlockHash.get(char(participentIDlist(p)))
        
        rawfname              = [rootDataSetPath, experimentName, '/', participentIDlist{p} '/block' num2str(b) '_raw.fif'];
        outfname              = [rootCodeOutputPath, version, '/', experimentName, '/1-preprosessing/sss/', participentIDlist{p} '_nkg_part' num2str(b) '_raw_sss_movecomp_tr.fif']; 
        logfname              = [rootCodeOutputPath, version, '/', experimentName, '/1-preprosessing/sss/', participentIDlist{p} '_nkg_part' num2str(b) '_raw_sss_movecomp_tr.log'];
        nameoffirstblock      = [rootDataSetPath, experimentName, '/', participentIDlist{p} '/block' num2str(1) '_raw.fif'];
                
        %RUN MAXFILTER 2.2 ('badstring ,... % the MEG bad channels' is missed out )
                              
        maxfiltercmd = ['/neuro/bin/util/maxfilter -f ' [rawfname] ' -o ' [outfname] ,...
        ' -ctc /neuro/databases/ctc/ct_sparse.fif ',...
        ' -cal /neuro/databases/sss/sss_cal.dat ',...
        ' -st 4 ',... % SSS with ST
        ' -corr 0.980 ',... % SSS with ST
        ' -linefreq 50 ',... % gets rid of mains frequency
        ' -autobad off ',...
        ' -trans ' [nameoffirstblock] ' ',... % transforms all blocks to same co-oridinates with respct to headpositions of the first block
        ' -lpfilt 35 ',...
        ' -frame head ',...
        ' -origin 0 0 45 ',... %Manual sphere z-coordinate: 55 mm for low-landmarks, 45 mm for high landmarks
        ' -hpistep 200 ',... % movement compensation
        ' -hpisubt amp ',... % movement compensation
        ' -movecomp ',...% movement compensation
        ' -hp ' [rawfname] '_hpi_movecomp.pos ',... % movement compensation
        ' -format short ',...
        ' -v | tee ' [logfname]
        ];

        eval([' ! ' maxfiltercmd ]);
        
        

        % Run trans %%%%%%%%%%%%%%%%%%%%%%%%%
    
    end
end
