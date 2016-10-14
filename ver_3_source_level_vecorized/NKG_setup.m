% setup file for neurokymatography scripts

rootPath = '/imaging/at03/NKG/';
savePath = '/imaging/at03/NKG/saved_data/source_data/';

participentIDlist = [
    320
    323
    324
    327
    348
    350
    363
    366
    371
    372
    377
    380
    397
    400
    401
    402
                    ];
              
%create hash table containing number of sessions
participentSessionHash = java.util.Hashtable;

for i = 1:numel(participentIDlist)
    eventfilename = ['/imaging/at03/NKG_Data_Sets/LexproMEG/meg08_0', num2str(participentIDlist(i)), '/meg08_0', num2str(participentIDlist(i)), '_part4-acceptedwordevents.eve'];
    if(exist(eventfilename, 'file'))
        participentSessionHash.put(num2str(participentIDlist(i)),4);
    else
        participentSessionHash.put(num2str(participentIDlist(i)),3);
    end
end

%create full wordlist (i.e. all words)
wordlistFilename = ['/imaging/at03/NKG_Data_Sets/LexproMEG/scripts/Simuli-Lexpro-MEG-Single-col.txt'];
fid = fopen(wordlistFilename);
wordlist = textscan(fid, '%s');
wordlist = wordlist{1};
fclose('all');