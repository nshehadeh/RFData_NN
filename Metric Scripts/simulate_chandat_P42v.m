function [] = simulate_chandat_P42v(useCodeFlag, code, L, SNR_dB)

% randomize random number generator
rng('shuffle')

% figure enables
show_fig = false;
show_mask = false;
save_fig = true;

% Transducer characteristics for phased array
decimate_factor = 5;
f0 = 2.72e6;
fs = 20 * f0;
c = 1540;                               % Speed of sound [m/s]
lambda = c / f0;                        % Wave length [m]
pitch = 300e-6;                         % array pitch
kerf = 50e-6;                           % Kerf [m]
element_width = 250e-6;                 % Width of element
element_height = 10*element_width;      % Height of element [m]
N_sub_x = 20;                           % number of mathematical elements in x direction
N_sub_y = 1;                            % Number of mathematical elements in y direction
N_elements = 64;                        % Number of elements in the transducer
Ncycles = 1;

% Coded excitation parameters
if ~useCodeFlag
    cexParams.code = 1;
    cexParams.useCodeFlag = 0;
    cexParams.L = [];
else
    cexParams.code = code;
    cexParams.useCodeFlag = 1;
    cexParams.L = L;
end
tChip = 0:1/fs:Ncycles/f0;  % time vector for chip waveform
cexParams.chip = sin(2*pi*f0*tChip);  % base pulse/chip waveform 
cexParams.Tp = length(cexParams.chip); 
cexParams.M = 300;
compoundCodeUpsamp = upsample(cexParams.code, cexParams.Tp);
compoundCodeUpsamp = compoundCodeUpsamp(1:end-cexParams.Tp+1);  % remove extra zeros at the end
cexParams.codedPulse = conv(cexParams.chip, compoundCodeUpsamp, 'full');
if cexParams.useCodeFlag
    cexParams.decodingFilter = genDecodingFilter(cexParams.code, cexParams.chip, cexParams.M, cexParams.L, cexParams.Tp);
else
    cexParams.decodingFilter = [];
end

% lateral extent of the transducer
aperture_width = N_elements * pitch;

% Set the transmit and recieve focus
focus = 10/100;                         % Transmit focus point [m]

% Find positions for each element of the array
element_position_x = (-(N_elements-1)/2:(N_elements-1)/2).'*pitch;
element_position_y = zeros(N_elements, 1);
element_position_z = zeros(N_elements, 1);

% transmit sequence
N_beams = 128; % 64
theta_start = -pi/4/4;
delta_theta = 2 * (-theta_start) / (N_beams - 1);
angles = (theta_start:delta_theta:-theta_start).';

% axial distance behind array for steered beams
apex = -aperture_width / 2 / tan(-theta_start);

% find transmit origin
TXorgs = [-apex * tan(angles), zeros(N_beams, 1), zeros(N_beams, 1)];

% find transmit focus
TXfocus = [sin(angles), zeros(N_beams, 1), cos(angles)];
TXfocus = TXorgs + focus * TXfocus;

%% transmit delays
transmit_delays = zeros(N_beams, N_elements);
H = -apex + focus;
for nbeam = 1:N_beams
    
    % set steering angle
    theta = angles(nbeam);
    
    % compute transmit delays
    delays = -sqrt( (element_position_x - TXfocus(nbeam, 1)).^2 + TXfocus(nbeam, 3)^2);
    delays = delays - min(delays);
    delays = delays / c;
    transmit_delays(nbeam, :) = delays.';
    
end

% set origin relative to timer which starts on first element fired
TXorgs_timer = zeros(N_beams, 3);
for nbeam = 1:N_beams
   [M, I] = min( transmit_delays(nbeam, :) );
   TXorgs_timer(nbeam, 1) = element_position_x(I);
end

%% Initialize field II

% Initialize FieldII
field_init(-1);

% set field parameters
set_field('c', c);                  % set speed of sound in tissue
set_field('fs', fs);                % set sampling frequency
set_field('show_times', 10)

% transmit aperture creation
transmit_aperture = xdc_linear_array(N_elements, element_width, element_height, kerf, N_sub_x, N_sub_y, [0 0 10]);

% transmit impulse response
impulse = cexParams.chip;
xdc_impulse(transmit_aperture, impulse)

% Set the excitation of the transmit aperture for coded excitation
excitation = cexParams.codedPulse;
xdc_excitation (transmit_aperture, excitation);

% receive aperture creation
receive_aperture = xdc_linear_array (N_elements, element_width, element_height, kerf, N_sub_x, N_sub_y, [0 0 10]);

% set receive aperture delays
xdc_focus_times( receive_aperture, 0, zeros(1, N_elements) )

% receive impulse
xdc_impulse(receive_aperture, impulse)

%% load the phantom
phantom_num = load('phantom_num.txt');
load(['../../sim_phantoms/phantom' num2str(phantom_num) '.mat'])

%% Perform the simulation

% allocate storage
chandat = zeros(1, N_elements, N_beams);
t0_list = zeros(N_beams, 1);
tic
for nbeam=1:N_beams
    
    % set transmit transducer focusing delays
    xdc_focus_times(transmit_aperture, 0, transmit_delays(nbeam, :));
    
    % Calculate the emitted pressure field
    [v, t0] = calc_scat_multi(transmit_aperture, receive_aperture, scat_positions, scat_amplitudes);

    % compute aperture data for center frequency
    t0_list(nbeam) = int64( t0 * fs );
    chandat(1:size(v, 1), :, nbeam) = v;

end
toc

% Free space for apertures
xdc_free (transmit_aperture)

% end field program
field_end();

%% pad chandat according to t0_list so that t=0 is first row of chandat
chandat_adjust = zeros(1, N_elements, N_beams);
for nbeam=1:N_beams
    v_temp = [zeros(t0_list(nbeam), N_elements); chandat(:, :, nbeam)];
    chandat_adjust(1:size(v_temp,1), :, nbeam) = v_temp;
end
chandat = chandat_adjust;
clear chandat_adjust;

%% adjust time vector based on the pulse length
% compute pulse length
pulse_length = length(excitation) + 2 * length(impulse) - 2;

%% Timer starts when first element fired. Adjust chandat so that
% t = 0 sample is when the wave goes by the transmit origin.

% Find time correction to adjust timing to start at origin
R_timer = sqrt( sum( (TXorgs_timer - TXfocus).^2, 2));
R_beam = sqrt( sum( (TXorgs - TXfocus).^2, 2));
t_correct = (R_timer - R_beam) / c * fs;

t_start = round( - pulse_length / 2 - t_correct);

% find smallest t0
pad_max = max(t_start - min(t_start));
chandat_adjust = zeros(1, N_elements, N_beams);
for nbeam=1:N_beams
    pad_top = t_start(nbeam) - min(t_start);
    pad_bottom = pad_max - pad_top;
    v_temp = [zeros(pad_top, N_elements); ...
                chandat(:, :, nbeam); ...
                zeros(pad_bottom, N_elements)];    
    chandat_adjust(1:size(v_temp,1), :, nbeam) = v_temp;
end
chandat = chandat_adjust;
clear chandat_adjust;
crop = max(t_start) - min(t_start);
chandat = chandat(1:end-crop, :, :);
N = size(chandat, 1);

% Apply coded excitation code compression, if applicable
if useCodeFlag
    chandat = codeCompress(chandat, cexParams.decodingFilter, fs);
end

% set time vector
t = ((0:size(chandat,1)-1)'+min(t_start));

% crop chandat
indx = find(t>=0, 1, 'first');
chandat = chandat(indx:end, :, :);
t = ( 0:1:(size(chandat, 1)-1) ).';

% downsample the data
chandat = chandat(1:decimate_factor:end, :, :);
fs = fs/decimate_factor;
t = ( 0:1:size(chandat, 1)-1).';

% crop chandat to phantom region
start = find(t/fs*c/2 >= ph.z_start*.9, 1, 'first');
stop = find(t/fs*c/2 < (ph.z_start+ph.z_size)*1.1, 1, 'last');
chandat = chandat(start:stop, :, :);
t = t(start:1:stop).';

%% apply dynamic receive focusing
chandat_beamformed = zeros(size(chandat));
for nbeam = 1:N_beams
    
    % find transmit angle
    theta = angles(nbeam);
    
    % find x position of wave at each point in time
    xf = (t / fs) * c / 2 * sin(theta);
    
    % find z position of wave at each point in time
    zf = (t / fs) * c / 2 * cos(theta);

    % find time when wave arrives at the point
    transmit_delay = (t / fs) / 2;
    
    % find element positions with respect to beam origin
    element_position_x_beam = element_position_x - TXorgs(nbeam, 1);

    for nrx = 1:N_elements
        
        % find receive delays
        receive_delay = sqrt(zf.^2 + (element_position_x_beam(nrx) - xf).^2) / c;
        
        % compute total delay
        total_delay = transmit_delay + receive_delay;
        
        % apply delays
        chandat_beamformed(:, nrx, nbeam) = interp1(t / fs, chandat(:, nrx, nbeam), total_delay, 'pchip', 0);
    end
    
end
chandat = chandat_beamformed;

% normalize the data
C = max(abs(chandat(:)));
chandat = chandat / C;

% ADD NOISE
if ~isinf(SNR_dB)
    noise = randn(size(chandat));
    noise_power = mean(noise(:).^2, 'omitnan');
    chandat_power = mean(chandat(:).^2, 'omitnan');
    C = chandat_power / 10^(SNR_dB/10);
    scalingParam = C / noise_power;
    chandat = chandat + noise * sqrt(scalingParam);
end

% Create rf data
rf_data = squeeze(sum(chandat, 2));
env = abs(hilbert(rf_data));
env = env / max(env(:));

% Create mask of data to use for training
depth = (t/fs)*c/2;
% X = zeros(length(depth), length(angles));
% Z = zeros(length(depth), length(angles));
% for nbeam = 1:length(angles)
%     theta = angles(nbeam);
%     X(:, nbeam) = sin(theta) * (depth - apex);
%     Z(:, nbeam) = cos(theta) * (depth - apex) + apex;
% end
% % create circular mask that includes the cyst and some surrounding area
% xc = ph.cystCenterX;
% zc = ph.cystCenterZ;
% radius_x = ph.cystRadius + 0.002;
% radius_z = ph.cystRadius + 0.002;
% mask = sqrt((X-xc).^2 / radius_x^2 + (Z-zc).^2 / radius_z^2) <= 1;

% create rectangular mask instead
tstart = find(depth >= ph.z_start, 1, 'first');
tstop = find(depth < (ph.z_start+ph.z_size), 1, 'last');
t_old = t; t = t(tstart:tstop);
xstart = 45; xstop = 85; % WARNING! This is hard coded and will need to change if the beams/angles are changed
angles_old = angles; angles = angles(xstart:xstop);
rf_data_old = rf_data; rf_data = rf_data(tstart:tstop, xstart:xstop);
env_old = env; env = env(tstart:tstop, xstart:xstop);

% save stuff
save('chandat.mat', 'chandat', 'rf_data', 'rf_data_old', 'env', 'env_old', ...
    'angles', 'angles_old', 't', 't_old', 'f0', 'fs', 'c', 'apex', 'cexParams', '-v7.3') 

% display image
if show_fig
    fig1 = figure('units','normalized','outerposition',[0 0 1 1]);
else
    fig1 = figure('units','normalized','outerposition',[0 0 1 1], 'Visible', 'off');
end
imagesc(20*log10(env), [-60, 0]), colormap(gray), colorbar
    
if save_fig
    print('das.png', '-dpng');        
end

% scan convert
min_phi = angles(1)*180/pi;
span_phi = (angles(end) - angles(1))*180/pi;
apex_cm = apex * 100;
dsfactor = 2;
vargin = [1e-10 0 5e-5 0 0 5e-5];   % [ ax_min ax_max ax_inc lat_min lat_max lat_inc ] (meters)

% pad the env
if t(1) > 1
    env = [zeros(t(1), length(angles)); env];   
elseif t(1) < 1
    idx = find(t >= 0, 1, 'first');
    env = env(idx:end, :);
end

[out,ax,la,outside] = scan_convert(env, 'sector', ...
    min_phi, span_phi, apex_cm, dsfactor, fs, vargin);
la = la / 100;
ax = ax / 100;
out = out / max(out(:));
    
if show_fig
    fig2 = figure('units','normalized','outerposition',[0 0 1 1]);
else
    fig2 = figure('units','normalized','outerposition',[0 0 1 1], 'Visible', 'off');
end
imagesc(la*1000, ax*1000, 20*log10(out), [-60 0])
colormap gray; axis image; colorbar
    
if save_fig
    print('das_sc.png', '-dpng');        
end
end
