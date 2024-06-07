%% modelo de filtro coclear
clear all

Fs = 44100;
freq = [100:300:3000]'; 
[fcoefs,~] = getFilterCoefs(freq,Fs);

y = 2*(rand(1,Fs)-0.5);
%plot(y)
%sound( [ zeros(1,Fs/10) , y] ,Fs)

yf = ERBFilterBank(y,fcoefs); 
%plot(yf')

%sound( [ zeros(1,Fs/10) , yf(10,:)] ,Fs)


%% transformada hilbert

yfh = NaN( size(yf) );
for f=1:length(freq)
    yfh(f,:) = hilbert( yf(f,:)  );
end

% clf
% plot(   yf(f,:) , 'r' )
% hold on
% %plot3( 1:length(yfh) , real(yfh) , imag(yfh) )
% plot(   abs(yfh) , 'b'  )
% plot(   angle(yfh)/pi / 100 , '.k' )



%% cipic hrtf database

azim = [-80 -65 -55 -45:5:45 55 65 80]';

temp = load('G:\My Drive\BICO\demo_FIacustico\cipic_subject_058\hrir_final.mat');

hrir_l = temp.hrir_l;
hrir_l = squeeze(hrir_l(:,9,:));
hrir_r = temp.hrir_r;
hrir_r=squeeze(hrir_r(:,9,:));

y_l = NaN( length(azim), length(y) );
y_r = NaN( length(azim), length(y) );

for a=1:length(azim)
    % plot(hrir_l(a,:), 'b')
    % hold on
    % plot(hrir_r(a,:), 'r')
    % title(num2str(azim(a)))
    y_l(a,:) = conv( y , hrir_l(a,:), 'same');
    y_r(a,:) = conv( y , hrir_r(a,:), 'same');
    %y_virt = [y_l' y_r'];
    %sound( [zeros(Fs/10,2);y_virt] ,Fs)
end


%% estimação de ITD e ILD (media e desvio, ao longo das frequencias)
clear all

% cria som
Fs = 44100;
%y = 2*(rand(1,4*Fs)-0.5); 
load('y.mat')

% funcao de transf da cabeça do volunt 58 do cipic
azim = [-80 -65 -55 -45:5:45 55 65 80]';
temp = load('G:\My Drive\BICO\demo_FIacustico\cipic_subject_058\hrir_final.mat');
hrir_l = temp.hrir_l;
hrir_l = squeeze(hrir_l(:,9,:));
hrir_r = temp.hrir_r;
hrir_r=squeeze(hrir_r(:,9,:));

%inicia filtros cocleares
freq = [400:100:1250]'; 
[fcoefs,~] = getFilterCoefs(freq,Fs);

%tira bordas da convolucao
idx_trim = 0.25*Fs : 3.75*Fs;

itd_media = NaN(length(freq),length(azim));
itd_desvio = NaN(length(freq),length(azim));

for a=1:length(azim)
    disp(num2str(a/length(azim)))
    
    %convolui som com funcao de transferencia
    y_l = conv( y , hrir_l(a,:), 'same');
    y_r = conv( y , hrir_r(a,:), 'same');    

    %aplica filtros cleares
    yf_l = ERBFilterBank(y_l,fcoefs); 
    yf_r = ERBFilterBank(y_r,fcoefs);     
   
    
    for f = 1:length(freq)
        %computa hilbert
        yfh_l = hilbert( yf_l(f,:) );
        yfh_r = hilbert( yf_r(f,:) );
        
        %computa ipd media e desvio
        ipd = wrapToPi( angle(yfh_r) - angle(yfh_l) );
        ipd_media = circ_mean(ipd(idx_trim)');
        ipd_desvio = circ_std(ipd(idx_trim)');
             
        %transforma em itd e corrige pra ambiguidade de fase (-pi = -3pi etc)
        itd_media(f,a) = ipd2itd( ipd_media , freq(f) );
        if f>1
            [ itd_media(f,a) , ~ ] = fixphase( itd_media(f,a),freq(f),itd_media(1,a) );
        end   
        itd_desvio(f,a) = ipd2itd( ipd_desvio , freq(f) );        
    end
end

subplot(131)
plot(azim,itd_media')
grid on
subplot(132)
plot(azim,itd_desvio')
grid on

% % Fisher information acustico (ITD)

AZIM = -80:1:80;

fit_option = fitoptions('Methods','SmoothingSpline','SmoothingParam',0.1);

FISHERINFO = NaN(length(freq),length(AZIM));
for f=1:length(freq)

    spline_flex = fit(azim,itd_media(f,:)','smoothingspline',fit_option);
    ITD_MEDIA = spline_flex(AZIM);
    ITD_MEDIADERIVADA = diff( spline_flex(-80.5:1:80.5) );

    spline_flex = fit(azim,itd_desvio(f,:)','smoothingspline',fit_option);
    ITD_DESVIO = spline_flex(AZIM);
    ITD_DESVIODERIVADA = diff( spline_flex(-80.5:1:80.5) );

    FISHERINFO(f,:) = (ITD_MEDIADERIVADA./ITD_DESVIO).^2 + 2*(ITD_DESVIODERIVADA./ITD_DESVIO).^2;
end

subplot(133)
plot(AZIM,FISHERINFO')
grid on
%%

