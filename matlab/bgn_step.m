function [terminated, vgi] = bgn_step(freq, amp, sim_time, noise_enabled)   
    load('bgn_vars.mat', 'pd','tmax','t','dt', 'n' ,'v1' ,'v2' ,'v3' ,'v4' ,'r' ,'Istim', 'Cm' ,'gl' ,'El' ,'gna' ,'Ena' ,'gk' ,'Ek' ,'gt' ,'Et' ,'gca' ,'Eca' ,'gahp' ,'k1' ,'kca' ,'A' ,'B' ,'the', 'gsyn', 'Esyn' ,'tau' ,'gpeak', 'gpeak1', 'vth' ,'vsn' ,'vge' ,'vgi' ,'S2' ,'S21' ,'S3' ,'S31' ,'S32','S4' ,'Z2' ,'Z4' ,'N2' ,'N3' ,'N4' ,'H1' ,'H2' ,'H3' ,'H4' ,'R1' ,'R2' ,'R3' ,'R4' ,'CA2' ,'CA3' ,'CA4', 'C2', 'i');
    freq = double(freq);
    amp = double(amp);
    
    % Default: noise off if not provided (backwards compatible)
    if nargin < 4
        noise_enabled = 0;
    end
    noise_enabled = double(noise_enabled);

    % to convert back to old impl, change freq --> action and amp --> 300
    dbs = zeros(1, sim_time);
    
    dt = 0.01;
    t = 0:dt:tmax;

    % --- Modified: make each pulse biphasic ---
    pulse_length = 0.3 / dt;
    half_len = round(pulse_length / 2);
    % First half positive, second half negative
    pulse = [amp * ones(1, half_len), -amp * ones(1, pulse_length - half_len)];

    j = 1;
    while j < length(t)
        dbs(j:j + pulse_length - 1) = pulse;
        instfreq = freq;
        isi = 1000 / instfreq;
        j = j + round(isi * (1 / dt));
    end

    sgis = zeros(n, sim_time);

    sim_step = 1;
    while sim_step <= sim_time && i <= tmax*100
        
        % ---- Per-timestep PD noise ----
        % When noise_enabled == 1, perturb pd at every timestep:
        %   pd_t = pd - rand_uniform(0.001, 0.01)
        % This simulates realistic brain environment fluctuations.
        % When noise_enabled == 0, pd_t = pd (original behaviour).
        if noise_enabled
            pd_t = max(0, pd - (0.001 + (0.01 - 0.001) * rand()));
        else
            pd_t = pd;
        end
        
        V1=vth(:,i-1);    V2=vsn(:,i-1);     V3=vge(:,i-1);    V4=vgi(:,i-1);
        % Synapse parameters 
        S21(2:n)=S2(1:n-1);S21(1)=S2(n);
        S31(1:n-1)=S3(2:n);S31(n)=S3(1);
        S32(3:n)=S3(1:n-2);S32(1:2)=S3(n-1:n);
        
        %membrane paremeters
        m1=th_minf(V1);m2=stn_minf(V2);m3=gpe_minf(V3);m4=gpe_minf(V4);
        n2=stn_ninf(V2);n3=gpe_ninf(V3);n4=gpe_ninf(V4);
        h1=th_hinf(V1);h2=stn_hinf(V2);h3=gpe_hinf(V3);h4=gpe_hinf(V4);
        p1=th_pinf(V1);
        a2=stn_ainf(V2); a3=gpe_ainf(V3);a4=gpe_ainf(V4);
        b2=stn_binf(R2);
        s3=gpe_sinf(V3);s4=gpe_sinf(V4);
        r1=th_rinf(V1);r2=stn_rinf(V2);r3=gpe_rinf(V3);r4=gpe_rinf(V4);
        c2=stn_cinf(V2);

        tn2=stn_taun(V2);tn3=gpe_taun(V3);tn4=gpe_taun(V4);
        th1=th_tauh(V1);th2=stn_tauh(V2);th3=gpe_tauh(V3);th4=gpe_tauh(V4);
        tr1=th_taur(V1);tr2=stn_taur(V2);tr3=30;tr4=30;
        tc2=stn_tauc(V2);
        
        %thalamic cell currents
        Il1=gl(1)*(V1-El(1));
        Ina1=gna(1)*(m1.^3).*H1.*(V1-Ena(1));
        Ik1=gk(1)*((0.75*(1-H1)).^4).*(V1-Ek(1));
        It1=gt(1)*(p1.^2).*R1.*(V1-Et);
        Igith=1.4*gsyn(6)*(V1-Esyn(6)).*S4; 
        
        %STN cell currents — uses pd_t instead of pd
        Il2=gl(2)*(V2-El(2));
        Ik2=gk(2)*(N2.^4).*(V2-Ek(2));
        Ina2=gna(2)*(m2.^3).*H2.*(V2-Ena(2));
        It2=gt(2)*(a2.^3).*(b2.^2).*(V2-Eca(2));
        Ica2=gca(2)*(C2.^2).*(V2-Eca(2));
        Iahp2=gahp(2)*(V2-Ek(2)).*(CA2./(CA2+k1(2)));
        Igesn=0.5*(gsyn(1)*(V2-Esyn(1)).*(S3+S31)); %Igesn=0;
        Iappstn=33-pd_t*10;
        
        %GPe cell currents — uses pd_t instead of pd
        Il3=gl(3)*(V3-El(3));
        Ik3=gk(3)*(N3.^4).*(V3-Ek(3));
        Ina3=gna(3)*(m3.^3).*H3.*(V3-Ena(3));
        It3=gt(3)*(a3.^3).*R3.*(V3-Eca(3));
        Ica3=gca(3)*(s3.^2).*(V3-Eca(3));
        Iahp3=gahp(3)*(V3-Ek(3)).*(CA3./(CA3+k1(3)));
        Isnge=0.5*(gsyn(2)*(V3-Esyn(2)).*(S2+S21)); %Isnge=0;
        Igege=0.5*(gsyn(3)*(V3-Esyn(3)).*(S31+S32)); %Igege=0;
        Iappgpe=21-13*pd_t+r;

        %GPi cell currents — uses pd_t instead of pd
        Il4=gl(3)*(V4-El(3));
        Ik4=gk(3)*(N4.^4).*(V4-Ek(3));
        Ina4=gna(3)*(m4.^3).*H4.*(V4-Ena(3));
        It4=gt(3)*(a4.^3).*R4.*(V4-Eca(3));
        Ica4=gca(3)*(s4.^2).*(V4-Eca(3));
        Iahp4=gahp(3)*(V4-Ek(3)).*(CA4./(CA4+k1(3)));
        Isngi=0.5*(gsyn(4)*(V4-Esyn(4)).*(S2+S21)); %Isngi=0;%special
        Igigi=0.5*(gsyn(5)*(V4-Esyn(5)).*(S31+S32)); %Igigi=0;%special
        Iappgpi=22-pd_t*6;
        
        %Differential Equations for cells
        %thalamic
        vth(:,i)= V1+dt*(1/Cm*(-Il1-Ik1-Ina1-It1-Igith+Istim(i)));
        H1=H1+dt*((h1-H1)./th1);
        R1=R1+dt*((r1-R1)./tr1);
        
        %STN
        vsn(:,i)=V2+dt*(1/Cm*(-Il2-Ik2-Ina2-It2-Ica2-Iahp2-Igesn+Iappstn+dbs(sim_step)));
        N2=N2+dt*(0.75*(n2-N2)./tn2); 
        H2=H2+dt*(0.75*(h2-H2)./th2);
        R2=R2+dt*(0.2*(r2-R2)./tr2);
        CA2=CA2+dt*(3.75*10^-5*(-Ica2-It2-kca(2)*CA2));
        C2=C2+dt*(0.08*(c2-C2)./tc2); 
        a=vsn(:,i-1)<-10 & vsn(:,i)>-10;
        u=zeros(n,1); u(a)=gpeak/(tau*exp(-1))/dt;
        S2=S2+dt*Z2; 
        zdot=u-2/tau*Z2-1/(tau^2)*S2;
        Z2=Z2+dt*zdot;
        
        %GPe
        vge(:,i)=V3+dt*(1/Cm*(-Il3-Ik3-Ina3-It3-Ica3-Iahp3-Isnge-Igege+Iappgpe));
        N3=N3+dt*(0.1*(n3-N3)./tn3);
        H3=H3+dt*(0.05*(h3-H3)./th3);
        R3=R3+dt*(1*(r3-R3)./tr3);
        CA3=CA3+dt*(1*10^-4*(-Ica3-It3-kca(3)*CA3));
        S3=S3+dt*(A(3)*(1-S3).*Hinf(V3-the(3))-B(3)*S3);
        
        %GPi
        vgi(:,i)=V4+dt*(1/Cm*(-Il4-Ik4-Ina4-It4-Ica4-Iahp4-Isngi-Igigi+Iappgpi));
        N4=N4+dt*(0.1*(n4-N4)./tn4);
        H4=H4+dt*(0.05*(h4-H4)./th4);
        R4=R4+dt*(1*(r4-R4)./tr4);
        CA4=CA4+dt*(1*10^-4*(-Ica4-It4-kca(3)*CA4));
        a=vgi(:,i-1)<-10 & vgi(:,i)>-10;
        u=zeros(n,1); u(a)=gpeak1/(tau*exp(-1))/dt;
        S4=S4+dt*Z4; 
        sgis(:, sim_step) = S4;
        zdot=u-2/tau*Z4-1/(tau^2)*S4;
        Z4=Z4+dt*zdot;
        sim_step = sim_step + 1;
        i = i + 1;

    end

    if i >= tmax*100
        terminated = 1;
    else 
        terminated = 0;
    end

    save('bgn_vars.mat', 'pd','tmax','t','dt', 'n' ,'v1' ,'v2' ,'v3' ,'v4' ,'r' ,'Istim', 'Cm' ,'gl' ,'El' ,'gna' ,'Ena' ,'gk' ,'Ek' ,'gt' ,'Et' ,'gca' ,'Eca' ,'gahp' ,'k1' ,'kca' ,'A' ,'B' ,'the', 'gsyn', 'Esyn' ,'tau' ,'gpeak', 'gpeak1', 'vth' ,'vsn' ,'vge' ,'vgi' ,'S2' ,'S21' ,'S3' ,'S31' ,'S32','S4' ,'Z2' ,'Z4' ,'N2' ,'N3' ,'N4' ,'H1' ,'H2' ,'H3' ,'H4' ,'R1' ,'R2' ,'R3' ,'R4' ,'CA2' ,'CA3' ,'CA4', 'C2', 'i', 'sgis','dbs');

end