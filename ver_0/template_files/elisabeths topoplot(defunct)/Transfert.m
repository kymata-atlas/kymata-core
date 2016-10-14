
addpath /opt/mne/matlab/toolbox/

clear all

a = fiff_read_evoked('matrix.fif')
load pvalues_test;
b = pvalues';

c = repmat(b, 1,240);
c(:,1201) = b(:,1);
c(306:420,:) = 0;
a.evoked.epochs(:,:) = c;

fiff_write_evoked('test.fif', a);

