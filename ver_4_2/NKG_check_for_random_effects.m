
leftright = 'rh';

results  = zeros(2562,201);

for time = 1:size(results,2)
    disp(['time:' num2str(time)]);
    parfor vertex = 1:size(results,1)
        disp(['vertex:' num2str(vertex)]);
        tvaluedistribution = [];
        for i = 1:length(participentIDlist)
            subjectname = char(participentIDlist(i));
            thisdata = load(['/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG/5-backedup-mat-files/vert2562-smooth5-nodepth-corFM-snr1-signed/randomeffeects/' subjectname '_TS-pitch_' leftright '_2562verts_-200-800ms_cuttoff600_15perms_ranksumpval_NegitiveIsZero_snr1_nodepth.mat'], 'outputSTC');
            tvaluedistribution = [tvaluedistribution thisdata.outputSTC.data((time*5)-4,vertex)];
        end
        [h,p] = ttest(tvaluedistribution);
        results(vertex, time) = p;
    end
end
