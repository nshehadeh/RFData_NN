function CNR = computeCNR(dataIn, dataOut)
    % don't consider negative pixels
    dataIn = nonzeros(dataIn);
    dataOut = nonzeros(dataOut);
    CNR = abs(mean(dataIn(:)) - mean(dataOut(:))) / sqrt(var(dataIn(:)) + var(dataOut(:)));
end