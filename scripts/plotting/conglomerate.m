
% deleate empty MYdata AND endTable

if isfield(mydata, leftright) ~= 0 
    it = length(mydata.(leftright).name)+1;
else
    it = 1;
end

results = outputSTC.data';
results = results(:,1:5:end,:);
results(:,202:end) = [];



if strcmp(leftright, 'lh')
    
    mydata.lh.data(:,:,it) = results;
    mydata.lh.name{it} = functionname;
    
    for i = 1:size(mydata.lh.data,1)
        
        vertmin = 1;
        vertlat = 0;
        
        disp(i);
        
        for j = 1:size(mydata.lh.data,3)
            
            [thismin, thislat] = min(mydata.lh.data(i,:,j));
            
            if thismin < vertmin
                vertmin = thismin;
                vertlat = thislat;
                vertfunctionname = mydata.lh.name(j);
            end
            
        end
        
        endtable.lh.vertices(i,1) =  vertmin;
        endtable.lh.vertices(i,2) =  vertlat;
        endtable.lh.name{i,1} =  vertfunctionname;
        
    end
    
else
    
    mydata.rh.data(:,:,it) = results;
    mydata.rh.name{it} = functionname;
    
    
    for  i = 1:size(mydata.rh.data,1)
        vertmin = 1;
        vertlat = 0;
        
        disp(i);
        
        for j = 1:size(mydata.rh.data,3)
            
            [thismin, thislat] = min(mydata.rh.data(i,:,j));
            
            if thismin < vertmin
                vertmin = thismin;
                vertlat = thislat;
                vertfunctionname = mydata.rh.name(j);
            end
            
        end
        
        endtable.rh.vertices(i,1) =  vertmin;
        endtable.rh.vertices(i,2) =  vertlat;
        endtable.rh.name{i,1} =  vertfunctionname;
    end
end