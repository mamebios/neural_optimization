function IPD = itd2ipd(ITD,FREQ)
%ITD in mus and FREQ in Hz to IPD in rad.
if max(size(FREQ))==1;
    FREQ=repmat(FREQ,size(ITD));
end
IPD = ((ITD*2*pi).*FREQ)/1000000;