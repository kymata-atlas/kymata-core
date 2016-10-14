leftright = 'lh';

result = [];
parfor i = 1:400
        disp(['seed:' num2str(i)]);
        thisdata = load(['/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG/5-backedup-mat-files/vert2562-smooth5-nodepth-corFM-snr1-signed/SnPM/seed' num2str(i) '_TS-pitch_' leftright '_2562verts_-200-800ms_cuttoff600_15perms_ranksumpval_NegitiveIsZero_snr1_nodepth.mat'], 'outputSTC');
        thisSTC = thisdata.outputSTC.data';
        thisSTC = thisSTC(:,1:5:end,:);
        thisSTC(:,202:end) = [];
        thisSTC(isnan(thisSTC)) = 1;%replace Nans with 1s (for excluded labels)
        result(i) = min(thisSTC(:));
end
threshold = prctile(result,5);

