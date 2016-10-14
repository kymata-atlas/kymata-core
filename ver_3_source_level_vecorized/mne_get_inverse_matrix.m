function [res] = mne_get_inverse_matrix(fname_inv,nave,lambda2,dSPM)
% modified from MNE's mne_ex_compute_inverse, in order to get the inverse matrix
% res.invmat contains inverse matrix
% dSPM only works for fixed orientations
% OH Feb 2010

% output structure res will contain inverse matrix in res.invmat

%fname_inv: filename of MNE inverse operator *inv.fif
%nave: number of (effective) averages, for dSPM only
%lambda2: regularization parameter (e.g. 1/3 for SNR=3, as is default in MNE)
%dSPM: if not empty, dSPM will be computed instead of current estimate

%% mne_ex_compute_inverse(fname_data,fname_inv,nave,lambda2)
%
% An example on how to compute a L2-norm inverse solution
% Actual code using these principles might be different because 
% the inverse operator is often reused across data sets.
%
%
% fname_data  - Name of the data file (not required for inverse matrix, OH)
% setno       - Data set number (not required for inverse matrix, OH)
% fname_inv   - Inverse operator file name
% nave        - Number of averages (scales the noise covariance) (for dSPM only, OH)
%               If negative, the number of averages in the data will be
%               used
% lambda2     - The regularization factor
% dSPM        - do dSPM?
%

%
%
%   Copyright 2006
%
%   Matti Hamalainen
%   Athinoula A. Martinos Center for Biomedical Imaging
%   Massachusetts General Hospital
%   Charlestown, MA, USA
%
%   No part of this program may be photocopied, reproduced,
%   or translated to another program language without the
%   prior written consent of the author.
%
%   $Id: mne_ex_compute_inverse.m 2623 2009-04-25 21:21:54Z msh $
%   
%   Revision 1.2  2008/05/26 10:49:26  msh
%   Update to incorporate the already weighted lead field basis
%
%   Revision 1.1  2006/05/05 03:50:40  msh
%   Added routines to compute L2-norm inverse solutions.
%   Added mne_write_inverse_sol_stc to write them in stc files
%   Several bug fixes in other files
%
%
%

me='MNE:mne_ex_compute_inverse';
FIFF=fiff_define_constants;

%if nargin ~= 4 % was 6 before OH
%   error(me,'Incorrect number of arguments'); 
%end
%
%   Read the data first
%
% data = fiff_read_evoked(fname_data,setno);
%
%   Then the inverse operator
%
inv = mne_read_inverse_operator(fname_inv);
%
%   Set up the inverse according to the parameters
%
try nave,
    if isempty(nave) || nave < 0
        nave = 100;
        % nave = data.evoked.nave;
    end
catch,
    nave = 100;
end;

try lambda2,
    if isempty(lambda2),
        lambda2 = 0;
    end;
catch
    lambda2 = 0;
end;

try dSPM,
catch
    dSPM = '';
end;

inv = mne_prepare_inverse_operator(inv,nave,lambda2,dSPM);
%
%   Pick the correct channels from the data
%
%data = fiff_pick_channels_evoked(data,inv.noise_cov.names);
%fprintf(1,'Picked %d channels from the data\n',data.info.nchan);
fprintf(1,'Computing inverse...');
%
%   Simple matrix multiplication followed by combination of the 
%   three current components
%
%   This does all the data transformations to compute the weights for the
%   eigenleads
%   
% trans =
% diag(sparse(inv.reginv))*inv.eigen_fields.data*inv.whitener*inv.proj*double(data.evoked(1).epochs); % as it was before OH
trans = diag(sparse(inv.reginv))*inv.eigen_fields.data*inv.whitener*inv.proj;
%
%   Transformation into current distributions by weighting the eigenleads
%   with the weights computed above
%
if isfield(inv, 'eigen_leads_weighted')
   %
   %     R^0.5 has been already factored in
   %
   fprintf(1,'(eigenleads already weighted)...');
   sol   = inv.eigen_leads.data*trans;
else
   %
   %     R^0.5 has to factored in
   %
   fprintf(1,'(eigenleads need to be weighted)...');
   sol   = diag(sparse(sqrt(inv.source_cov.data)))*inv.eigen_leads.data*trans;
end
   
% if inv.source_ori == FIFF.FIFFV_MNE_FREE_ORI
%     fprintf(1,'combining the current components...');
%     sol1 = zeros(size(sol,1)/3,size(sol,2));
%     for k = 1:size(sol,2)
%         sol1(:,k) = sqrt(mne_combine_xyz(sol(:,k)));
%     end
%     sol = sol1;
% end
if dSPM % at the moment this only works for fixed orientations
    fprintf(1,'(dSPM)...');
    sol = inv.noisenorm*sol;
end

res.invmat = sol;
res.inv = inv;

% as it was before OH:
% res.inv   = inv;
% res.sol   = sol;
% res.tmin  = double(data.evoked(1).first)/data.info.sfreq;
% res.tstep = 1/data.info.sfreq;
% fprintf(1,'[done]\n');

return;
end

