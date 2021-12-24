function R = calculateBeamformedTemporalLOC(RFdata, W)
    halfW = ceil(W/2);
    N_ax = size(RFdata, 1); 
    N_beam = size(RFdata, 2);
    R = zeros(N_ax, N_beam);
    for m = 1:N_ax
        zidx = max(1, m-halfW):min(N_ax, m+halfW);
        for k = 1:N_beam
            data_a = squeeze(RFdata(zidx, k, 1:end-1));  % ignoring imag for quick sim
            data_b = squeeze(RFdata(zidx, k, 2:end));
            num = sum(data_a(:) .* conj(data_b(:)));
            denom = sqrt(sum(data_a(:) .* conj(data_a(:))) * sum(data_b(:) .* conj(data_b(:))));
            R(m, k) = num/denom;
        end
    end
end
