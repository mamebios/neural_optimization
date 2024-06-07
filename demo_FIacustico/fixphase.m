function [ fixed_itd , fixed_ipd ] = fixphase( itd,freq,correct_itd )
%FIXPHASE Summary of this function goes here
%   [ fixed_itd , fixed_ipd ] = fixphase( itd,freq,correct_itd )
    correct_ipd = itd2ipd(correct_itd,freq);
    ipd = itd2ipd(itd,freq);
    ipd_candidates = ipd + (-20:20)*(2*pi);
    [~,idx]=min(abs(ipd_candidates-correct_ipd));
    fixed_ipd = ipd_candidates(idx);
    fixed_itd = ipd2itd(fixed_ipd,freq);
end

