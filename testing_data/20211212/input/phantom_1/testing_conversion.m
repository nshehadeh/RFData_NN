% This code snippet loads in the enveloped data (abs(hilbert(chandat)) 
% and performs scan conversion which converts the native polar sampling of
% the image to rectangular/cartesian sampling for more accurate display. 
% This relies on the function src/scan_convert.

load('chandat.mat', 'angles', 'apex', 'env', 'fs')


% scan convert
min_phi = angles(1)*180/pi;
span_phi = (angles(end) - angles(1))*180/pi;
apex_cm = apex * 100;
dsfactor = 2;
vargin = [1e-10 0 5e-5 0 0 5e-5]; % (meters)

t(1)
length(angles)
% pad the env
if t(1) > 1
    env = [zeros(t(1), length(angles)); env];   
elseif t(1) < 1
    idx = find(t >= 0, 1, 'first');
    env = env(idx:end, :);
end

[out, ax, la, outside] = scan_convert(env, 'sector', min_phi, span_phi, apex_cm, dsfactor, fs, vargin);
la = la/100;
ax = ax/100;
out = out/max(out(:));
    
figure, imagesc(la*1000, ax*1000, 20*log10(out), [-60 0])
colormap gray; axis image; colorbar
