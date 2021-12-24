function C  = computeContrast(dataIn, dataOut)
    % don't consider negative pixels
    dataIn = nonzeros(dataIn);
    dataOut = nonzeros(dataOut);
    C = mean(dataIn(:)) / mean(dataOut(:));
end