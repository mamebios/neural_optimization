function [fcoefs,ERB]=getFilterCoefs(CF,FS)

%Check for Hz
if max(CF)<100
    display('Error: CF should be in Hz')
end


%Number of frequency channels
NF=length(CF);


%Q10 according to barn owl data
Q10=.074*CF.^(.5);

%Get filter coefficients
fcoefs=zeros(NF,10);
for n=1:NF
    cf=[CF(n) 20000];
    %temp= MakeERBFilters(30000,cf,100,Q10(n));
    [temp,ERB(n,:)] = MakeERBFilters(FS,cf,100,Q10(n));
    fcoefs(n,:)=temp(1,:); 
end