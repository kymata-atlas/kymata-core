%----------------
% create stimuli signals .mat file
%----------------
 
stimulisig = [];
stimulisigIDfilenames = '/imaging/at03/Method_to_locate_neurocorrelates_of_processes/signals/pitch/aa_names.txt';



fid = fopen(stimulisigIDfilenames);
    names = textscan(fid, '%s');
fclose(fid);

for n = 1:length(names{1,1})
    thisname = char(names{1,1}(n,1));
    stimulisigIDfilename = ['/imaging/at03/Method_to_locate_neurocorrelates_of_processes/signals/pitch/', thisname, '.txt'];
    tempdata = importdata(stimulisigIDfilename,'\t',1);
    stimulisig.name{n,1} = thisname;
    for i=1: size(tempdata.data,2)
            stimulisig.time(n,i) = tempdata.data(1,i);
            stimulisig.pitch(n,i) = tempdata.data(2,i); % should read in all the signals here
            %stimulisig.deltaC0(n,i) = tempdata.data(3,i); % should read in all the signals here
            %stimulisig.deltadeltaC0(n,i) = tempdata.data(4,i); % should read in all the signals here
    end
end