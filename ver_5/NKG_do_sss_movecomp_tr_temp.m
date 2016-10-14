%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% run raw data through maxfilter 2.2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% based partly on Jason's script and partly on Elisabeth's.

for p = 1:numel(participentIDlist) 
               
    for b = 1:participentNumBlockHash.get(char(participentIDlist(p)))
        
        rawfname    = [rootDataSetPath, 'DATASET_1-01_visual-only/', participentIDlist{p} '/', participentIDlist{p}, '_nkg_part' num2str(b) '_raw'];
        outfname    = [rootCodeOutputPath, version, '/', experimentName, '/1-preprosessing/sss/', participentIDlist{p} '_nkg_part' num2str(b) '_raw_sss_movecomp.fif']; 
        logfname    = [rootCodeOutputPath, version, '/', experimentName, '/1-preprosessing/sss/', participentIDlist{p} '_nkg_part' num2str(b) '_raw_sss_movecomp.log'];
              
        %GET MEG BAD CHANNELS (missed out as already marked in previous step)
        
        %fid = fopen(badEMEGfname);
        
        %badstring = ' -bad ';
    
        %if fid == 3
        %    badchannels = textscan(fid, '%s %d', 'delimiter', ' ');
        %    %if no file, or empty then no bad channels
        %
        %        for i = 1:length(badchannels{1,2})
        %            if (strcmp(badchannels{1,1}(i),'MEG'))
        %                badstring = [badstring num2str(badchannels{1,2}(i)) ' '];
        %            end
        %        end
        %    end
        
        fclose('all');
        
        %disp(badstring);
        
        %RUN MAXFILTER 2.2 ('badstring ,... % the MEG bad channels' is missed out )
                              
        maxfiltercmd = ['/neuro/bin/util/maxfilter -f ' [rawfname, '.fif'] ' -o ' [outfname] ,...
        ' -ctc /neuro/databases/ctc/ct_sparse.fif ',...
        ' -cal /neuro/databases/sss/sss_cal.dat ',...
        ' -linefreq 50 ',... % gets rid of mains frequency
        ' -autobad off ',...
        ' -st 4 ',... % SSS with ST
        ' -corr 0.980 ',... % SSS with ST
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
        
        
        % (4) %%%%%%%%%%%%%%%%%%%%%%%%%


%         % TRANSFORMATION (all but first file, block 1):
%         if bb>1
%         
%             trcmd=sprintf(' -trans %s -frame head -origin 0 0 45',b1file);
% 
%             mfcmd4=[
%             '/neuro/bin/util/maxfilter -f ' [outfname2] ' -o ' [outfname3],...
%             '  -autobad off ', trcmd, ' -force -v | tee ' logfname3
%             ];
%             fprintf(1, '\nMaxfiltering... -trans\n');
%             fprintf(1, '%s\n', mfcmd4);
%             eval([' ! ' mfcmd4 ])
%             
%         else
%             
%             saveas tr
%             
%         end
        
    end
    
end
