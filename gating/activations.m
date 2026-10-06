function ainf=gpe_ainf(V)
    ainf=1./(1+exp(-(V+57)./2));
    return

function hinf=gpe_hinf(V)
    hinf=1./(1+exp((V+58)./12));
    return

function minf=gpe_minf(V)
    minf=1./(1+exp(-(V+37)./10));
    return

function ninf=gpe_ninf(V)
    ninf=1./(1+exp(-(V+50)./14));
    return

function rinf=gpe_rinf(V)
    rinf=1./(1+exp((V+70)./2));
    return

function sinf=gpe_sinf(V)
    sinf=1./(1+exp(-(V+35)./2));
    return

function tau=gpe_tauh(V)
    tau=0.05+0.27./(1+exp(-(V+40)./-12));
    return


function tau=gpe_taun(V)
    tau=0.05+0.27./(1+exp(-(V+40)./-12));
    return

function h=Hinf(V)
    h=1./(1+exp(-(V+57)./2));
    return

function ainf=stn_ainf(V)
    ainf=1./(1+exp(-(V+63)./7.8));
    return

function binf=stn_binf(R)
    binf=1./(1+exp(-(R-0.4)./0.1))-1/(1+exp(0.4/0.1));
    return


function cinf=stn_cinf(V)
    cinf=1./(1+exp(-(V+20)/8));
    return

function hinf=stn_hinf(V)
    hinf=1./(1+exp((V+39)./3.1));
    return

function minf=stn_minf(V)
    minf=1./(1+exp(-(V+30)./15));
    return


function ninf=stn_ninf(V)
    ninf=1./(1+exp(-(V+32)./8.0));
    return

function rinf=stn_rinf(V)
    rinf=1./(1+exp((V+67)./2));
    return

function sinf=stn_sinf(V)
    sinf=1./(1+exp(-(V+39)./8));
    return


function tau=stn_tauc(V)
    tau=1+10./(1+exp((V+80)/26));
    return

function tau=stn_tauh(V)
    tau=1+500./(1+exp(-(V+57)./-3));
    return

function tau=stn_taun(V)
    tau=1+100./(1+exp(-(V+80)./-26));
    return

function tau=stn_taur(V)
    tau=7.1+17.5./(1+exp(-(V-68)./-2.2));
    return

function hinf=th_hinf(V)
    hinf=1./(1+exp((V+41)./4));
    return


function minf=th_minf(V)
    minf=1./(1+exp(-(V+37)./7));
    return

function pinf=th_pinf(V)
    pinf=1./(1+exp(-(V+60)./6.2));
    return

function rinf=th_rinf(V)
    rinf=1./(1+exp((V+84)./4));
    return


function tau=th_tauh(V)
    tau=1./(ah(V)+bh(V));

function a=ah(V)
    a=0.128*exp(-(V+46)./18);
    
function b=bh(V)
    b=4./(1+exp(-(V+23)./5));
    return

function tau=th_taur(V)
    tau=0.15*(28+exp(-(V+25)./10.5));
    return




















































