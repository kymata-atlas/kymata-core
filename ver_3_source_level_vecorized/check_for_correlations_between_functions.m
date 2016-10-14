% for pop one - pop 2


vertex = 310;
time = 85;
hemisphere = 'lh';
x = 'EMEG';
y = 'pitch';
z = 'loudness';

xytakeoutzDistribution = [];

rhoxy = ;
rhoxz = ;
rhoyz = ;

for i = 1:numel(xytakeoutzDistribution) 
     
  partialcorrelation = (rhoxy(i) - (rhoxz(i) - rhoyz(i)))/(sqrt(1-(rhoxz(i)^2))*sqrt(1-(rhoyz(i)^2)));
  
  xytakeoutzDistribution = [xytakeoutzDistribution partialcorrelation];
  
end 
