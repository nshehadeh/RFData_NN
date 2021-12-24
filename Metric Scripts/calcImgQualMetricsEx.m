%% Load in Data and Define ROIs
clear
load('input_old', 't', 'fs', 'c', 'angles', 'apex', 'env')
load('new_data4')
env = permute(new_env, [2, 1]);
load('phantom1.mat', 'ph')

depth = (t/fs)*c/2;
X = zeros(length(depth), length(angles));
Z = zeros(length(depth), length(angles));
for nbeam = 1:length(angles)
    theta = angles(nbeam);
    X(:, nbeam) = sin(theta) * (depth - apex);
    Z(:, nbeam) = cos(theta) * (depth - apex) + apex;
end
xc = ph.cystCenterX;
zc = ph.cystCenterZ;
radius_x = ph.cystRadius - 0.001;
radius_z = ph.cystRadius - 0.001;
mask_in = sqrt((X-xc).^2 / radius_x^2 + (Z-zc).^2 / radius_z^2) <= 1;

radius_x_outer = ph.cystRadius + 0.0018;
radius_z_outer = ph.cystRadius + 0.0018;
radius_x_inner = ph.cystRadius + 0.0005;
radius_z_inner = ph.cystRadius + 0.0005;
mask_outer = sqrt((X-xc).^2 / radius_x_outer^2 + (Z-zc).^2 / radius_z_outer^2) <= 1;
mask_inner = sqrt((X-xc).^2 / radius_x_inner^2 + (Z-zc).^2 / radius_z_inner^2) <= 1;
mask_outer(mask_inner) = 0;
mask_out = mask_outer; clear mask_outer; clear mask_inner;

% Verify inside and outside mask locations and that they have approximately
% the same number of pixels 
% figure, imagesc(mask_in + 2*mask_out)
% length(find(mask_in))
% length(find(mask_out))
% figure, imagesc(mask_in.*(20*log10(env))), colormap gray
% figure, imagesc(mask_out.*(20*log10(env))), colormap gray

%% Compue Image Quality Metrics

% compute contrast to noise ratio
CNR = 20*log10(computeCNR(env.*mask_in, env.*mask_out)) % units of dB
% The maximum theoretical CNR is 5.6 dB for speckle, so you want this
% number to be as high as possible (but it shouldn't exceed 5.6 dB)

% compute contrast ratio
CR = -20*log10(computeContrast(env.*mask_in, env.*mask_out)) % units of dB (negative sign is convention, not super important)
% Since this lesion is supposed to be anechoic, the true contrast ratio is 
% infinity. Clutter in the cyst reduces the contrast. You want this number
% to be as large as possible. However, there is often a trade-off between
% high CR and high CNR/

% compute speckle signal to noise ratio
% this is slightly different than normal SNR; this is an indicator of the
% quality of the texture of the speckle background
SNRs = 20*log10(computeSNRs(env.*mask_out))
% You also want this number to be as high as possible.
