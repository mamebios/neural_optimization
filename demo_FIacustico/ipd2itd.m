function ITD = ipd2itd(IPD,FREQ)
% ITD = ipd2itd(IPD,FREQ)
% IPD in rad and FREQ in Hz to ITD in mus.
if max(size(FREQ))==1;
    FREQ=repmat(FREQ,size(IPD));
end
ITD = (IPD*1000000)./(FREQ*2*pi);