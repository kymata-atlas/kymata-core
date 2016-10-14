

%Copy elisabeths forward files (done)
    
xxx
xxx
xxx

%try to duplicate her old file from her foward and her niose cov.   
%the old file is ['/imaging/ef02/lexpro/meg08_0' num2str(participentIDlist(p)) '/tr/meg08_0' num2str(participentIDlist(p)) '_3L-loose0.2-depth-reg-inv.fif']); % change to mine      

%where is her origioanl do inverse file?

unixCommand = ['mne_do_inverse_operator '];
unixCommand = [unixCommand '--fwd /imaging/at03/?????????????? '];
%unixCommand = [unixCommand '--fixed  ' ];
unixCommand = [unixCommand '--loose 0.2  ' ];
unixCommand = [unixCommand '--depth ???  ' ];
unixCommand = [unixCommand '--bad ???  ' ];
unixCommand = [unixCommand '--sensecov ???  ' ];
unixCommand = [unixCommand '--megreg ???  ' ];
unixCommand = [unixCommand '--eegreg ??? ' ];
unixCommand = [unixCommand '--fmrithresh 1 ']; %manditory but ignored
unixCommand = [unixCommand '--inv ???? '];
fprintf(['[unix:] ' unixCommand '\n']);
unix(unixCommand);


%done

%Create a new covarience from her foward and empty noise cov.

unixCommand = ['mne_do_inverse_operator '];
unixCommand = [unixCommand '--fwd /imaging/at03/?????????????? '];
%unixCommand = [unixCommand '--fixed  ' ];
unixCommand = [unixCommand '--loose 0.2  ' ];
unixCommand = [unixCommand '--depth ???  ' ];
unixCommand = [unixCommand '--bad ???  ' ];
unixCommand = [unixCommand '--sensecov ???  ' ];
unixCommand = [unixCommand '--megreg ???  ' ];
unixCommand = [unixCommand '--eegreg ??? ' ];
unixCommand = [unixCommand '--fmrithresh 1 ']; %manditory but ignored
unixCommand = [unixCommand '--inv ???? '];
fprintf(['[unix:] ' unixCommand '\n']);
unix(unixCommand);


