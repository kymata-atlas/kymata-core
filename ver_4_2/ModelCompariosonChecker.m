% Basic model comparison.
    

inputfolder = '/imaging/at03/NKG_Code_output/Version4_2/VerbphraseMEG/5-backedup-mat-files/';

phonslh = load([inputfolder, 'averagemesh-vert2562-smooth5-elisinvsol/GM-loudness-instant-phons-NegitiveIsZero_2562verts_snr1_nodepth-200-800ms_cuttoff1000_8perms_lh_ttest2pval.mat'],'outputSTC');
phonsrh = load([inputfolder, 'averagemesh-vert2562-smooth5-elisinvsol/GM-loudness-instant-phons-NegitiveIsZero_2562verts_snr1_nodepth-200-800ms_cuttoff1000_5perms_rh_ttest2pval.mat'],'outputSTC');
soneslh = load([inputfolder, 'averagemesh-vert2562-smooth5-elisinvsol/GM-loudness-instant-sones-NegitiveIsZero_2562verts_snr1_nodepth-200-800ms_cuttoff1000_8perms_lh_ttest2pval.mat'],'outputSTC');
sonesrh = load([inputfolder, 'averagemesh-vert2562-smooth5-elisinvsol/GM-loudness-instant-sones-NegitiveIsZero_2562verts_snr1_nodepth-200-800ms_cuttoff1000_8perms_rh_ttest2pval.mat'],'outputSTC');

phonslh = phonslh.outputSTC.data';
phonsrh = phonsrh.outputSTC.data';
soneslh = soneslh.outputSTC.data';
sonesrh = sonesrh.outputSTC.data';

phonslh = phonslh(:,1:5:end,:);
phonsrh = phonsrh(:,1:5:end,:);
soneslh = soneslh(:,1:5:end,:);
sonesrh = sonesrh(:,1:5:end,:);

phonslh(:,202:end) = [];
phonsrh(:,202:end) = [];
soneslh(:,202:end) = [];
sonesrh(:,202:end) = [];

phonslh(isnan(phonslh)) = 1;
phonsrh(isnan(phonsrh)) = 1;
soneslh(isnan(soneslh))  = 1;
sonesrh(isnan(sonesrh)) = 1;

phonslh = min(phonslh,[],2);
phonsrh = min(phonsrh,[],2);
soneslh = min(soneslh,[],2);
sonesrh = min(sonesrh,[],2);

phons = [phonslh' phonsrh'];
sones = [soneslh' sonesrh'];

[p, h, stats] = signrank(sones,phons,'tail','left');