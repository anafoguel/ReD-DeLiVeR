import math
import src.pars as par


def realPi(r) :
    fvthr = 1.666666666666667e0
    rmax  = 1.e6
    # use assymptotic formula
    if ( abs(r)<1e-3 ) :
        return -fvthr-math.log(r)
    # return zero for large values
    elif(abs(r)>rmax) :
        return 0.;
    elif(4.*r>1.) :
        beta=math.sqrt(4.*r-1.)
        return 1./3. -(1.+2.*r)*(2.-beta*math.acos(1.-1./(2.*r)))
    else :
        beta=math.sqrt(1.-4.*r)
        return 1./3.-(1.+2.*r)*(2.+beta*math.log(abs((beta-1.)/(beta+1.))))
    
def alphaEM(scale) :
    eps=1e-6
    a1=0.0
    b1=0.00835
    c1=1.000
    a2=0.0    
    b2=0.00238
    c2=3.927
    a3=0.00165
    b3=0.00299
    c3=1.000
    a4=0.00221
    b4=0.00293
    c4=1.000
    # alpha_EM at Q^2=0
    alem=7.2973525693e-3
    aempi = alem/(3.*math.pi)
    # return q^2=0 value for small scales
    if(scale<eps) :
        return alem
    # leptonic component
    repigg = aempi*(realPi(par.me_**2/scale)+realPi(par.mmu_**2/scale)+realPi(par.mtau_**2/scale))
    # Hadronic component from light quarks
    if(scale<9e-2) :
        repigg+=a1+b1*math.log(1.+c1*scale)
    elif(scale<9.) :
        repigg+=a2+b2*math.log(1.+c2*scale);
    elif(scale<1.e4) :
        repigg+=a3+b3*math.log(1.+c3*scale);
    else :
        repigg+=a4+b4*math.log(1.+c4*scale);
    # Top Contribution
    repigg+=aempi*realPi(par.mt_**2/scale);
    # return the answer
    return alem/(1.-repigg);



def alphaQCD(Q):
    if Q<par.mc_:
        nf=3
    if par.mc_<=Q<=par.mb_:
        nf = 4
    if par.mb_<Q<=par.mt_:
        nf= 5
    if Q>par.mt_:
        nf=6
    b0 = 11 - (2/3)*nf
    b1= 102 - (38/3)*nf
    b2=(2857/2)-(5033/18)*nf + (325/54)*nf**2
    b3=100541 - 24423.3*nf + 1625.4*nf**2 - 27.493*nf**3
    Lambda =  0.212
    logS = math.log(Q**2/Lambda**2)
    pre = 4*math.pi/b0/logS
    coef1 = -b1*math.log(logS)/b0**2/logS
    coef2 = (b1**2/b0**4/logS**2)*(math.log(logS)**2- math.log(logS) - 1 + (b2*b0/b1**2))
    coef3 = (b1**3/b0**6/logS**3)*(-(math.log(logS))**3 + (5/2)*(math.log(logS))**2 + 2* math.log(logS) - (1/2) - 3*(b2*b0/b1**2)*math.log(logS)+(b3*b0**2/2/b1**3))
    return pre*(1+coef1+coef2+coef3)
