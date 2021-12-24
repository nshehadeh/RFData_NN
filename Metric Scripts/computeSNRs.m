function SNRs = computeSNRs(dataOut)
    % don't consider negative pixels
    dataOut = nonzeros(dataOut);
    SNRs =  mean(dataOut(:))/std(dataOut(:));
end