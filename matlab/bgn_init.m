function bgn_init(pd,tmax,seed)
    tmax = double(tmax);
    
    % Default pd=1 (full PD) if not provided or empty
    if nargin < 1 || isempty(pd)
        pd = 1;
    end
    pd = double(pd);
    
    % Seed is optional — if not provided, MATLAB uses its own RNG state
    % Nick's code calls bgn_init(pd, tmax) with no seed
    % Our MAB code calls bgn_init(pd, tmax, seed) with explicit seed
    if nargin >= 3 && ~isempty(seed)
        seed = double(seed);
        rng(seed);
    end
    
    dt=0.01; %timestep (ms)
    t=0:dt:tmax;
    n=10; 
    
    v1=-62+randn(n,1)*5;
    v2=-62+randn(n,1)*5;
    v3=-62+randn(n,1)*5;
    v4=-62+randn(n,1)*5;
    r=randn(n,1)*2;
    
    Istim=zeros(1,length(t));
    deltasm=5;
    pulse=3.5*ones(1,deltasm/dt);
    smc_freq=14;
    cv=0.2;
    i=1; j=1;
    A = 1/cv^2;
    B  = smc_freq / A;
    if cv==0
        instfreq=smc_freq;
    else
        instfreq=gamrnd(A,B);
    end
    ipi=1000/instfreq;
    i=i+round(ipi/dt);
    while i+500<length(t)
        Istim(i:i+deltasm/dt-1)=pulse;
        A = 1/cv^2;
        B  = smc_freq / A;
        if cv==0
            instfreq=smc_freq;
        else
        instfreq=gamrnd(A,B);
        end
        ipi=1000/instfreq;
        i=i+round(ipi/dt);
        j=j+1;
    end
    
    
    addpath("gating");
    
    Cm=1;
    gl=[0.05 2.25 0.1]; El=[-70 -60 -65];
    gna=[3 37 120]; Ena=[50 55 55]; 
    gk=[5 45 30]; Ek=[-75 -80 -80];
    gt=[5 0.5 0.5]; Et=0;
    gca=[0 2 0.15]; Eca=[0 140 120];
    gahp=[0 20 10]; k1=[0 15 10]; kca=[0 22.5 15];
    A=[0 3 2 2]; B=[0 0.1 0.04 0.04]; the=[0 30 20 20]; 
    
    
    gsyn = [1 0.3 1 0.3 1 .08]; Esyn = [-85 0 -85 0 -85 -85];
    tau=5; gpeak=0.43;gpeak1=0.3;  
    
    vth=zeros(n,length(t)); %thalamic membrane voltage
    vsn=zeros(n,length(t)); %STN membrane voltage
    vge=zeros(n,length(t)); %GPe membrane voltage
    vgi=zeros(n,length(t)); %GPi membrane voltage
    S2=zeros(n,1); S21=zeros(n,1); S3=zeros(n,1); 
    S31=zeros(n,1);S32=zeros(n,1); S4=zeros(n,1); 
    Z2=zeros(n,1);Z4=zeros(n,1);
    vth(:,1)=v1;
    vsn(:,1)=v2;
    vge(:,1)=v3;
    vgi(:,1)=v4;
    
    N2=stn_ninf(vsn(:,1)); N3=gpe_ninf(vge(:,1));N4=gpe_ninf(vgi(:,1));
    H1=th_hinf(vth(:,1)); H2=stn_hinf(vsn(:,1)); H3=gpe_hinf(vge(:,1));H4=gpe_hinf(vgi(:,1));
    R1=th_rinf(vth(:,1)); R2=stn_rinf(vsn(:,1)); R3=gpe_rinf(vge(:,1));R4=gpe_rinf(vgi(:,1));
    CA2=zeros(10,1); CA3=CA2;CA4=CA2; 
    C2=stn_cinf(vsn(:,1));
    i=2;
    save('bgn_vars.mat', 'pd','tmax', 't','dt', 'n' ,'v1' ,'v2' ,'v3' ,'v4' ,'r' ,'Istim', 'Cm' ,'gl' ,'El' ,'gna' ,'Ena' ,'gk' ,'Ek' ,'gt' ,'Et' ,'gca' ,'Eca' ,'gahp' ,'k1' ,'kca' ,'A' ,'B' ,'the', 'gsyn', 'Esyn' ,'tau' ,'gpeak', 'gpeak1', 'vth' ,'vsn' ,'vge' ,'vgi' ,'S2' ,'S21' ,'S3' ,'S31' ,'S32','S4' ,'Z2' ,'Z4' ,'N2' ,'N3' ,'N4' ,'H1' ,'H2' ,'H3' ,'H4' ,'R1' ,'R2' ,'R3' ,'R4' ,'CA2' ,'CA3' ,'CA4', 'C2', 'i');
    
end