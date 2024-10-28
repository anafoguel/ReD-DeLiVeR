# Library to load
import math,cmath
import scipy.integrate
from src.functions import alpha,Resonance
import src.pars as par

#define complex number
ii=complex(0,1)


#
####################################################################################
## Parameter set for hadronic current, own parametrization                        ##
####################################################################################
#
# masses and width from PDG

#mPi0, mPi+-
mPi_ = [0.1349768, 0.13957039 ]
#mK+-, mK0
mK_ = [0.493677, 0.497611]
#mK*(890)
mKstar_ = 0.89166

# KKpipi parametrization

# masses and amplitudes, I=0, phi resonances (mode 0)

isoScalarMasses0  = [1019.461*par.MeV, 1650*par.MeV,1957*par.MeV]
isoScalarWidths0  = [   4.249*par.MeV, 0.10306362864702914,  267*par.MeV]
isoScalarAmp0   = [0.  ,4.5159727286986975 ,0.  ]
isoScalarPhase0 = [0., math.pi,0.]

# masses and amplitudes, I=1, rho resonances (mode 0)

isoVectorMasses0 = [775.26*par.MeV,1470*par.MeV,1.8424500809807751]
isoVectorWidths0 = [149.1 *par.MeV, 400*par.MeV, 0.40316760691226605]
isoVectorAmp0 = [0. ,0. , 15.658829780911248 ]
isoVectorPhase0 = [0,math.pi,0.]

# masses and amplitudes, I=0, phi resonances (mode 1,2,3)

isoScalarMasses  = [1019.461*par.MeV, 1.7 , 1.996778573169961]
isoScalarWidths  = [   4.249*par.MeV, 0.3,  0.6014411030707548]
isoScalarAmp   = [0.  ,2.488597350435909 , 0.]
isoScalarPhase = [0., 1.0180585878258477 ,0.]

# masses and amplitudes, I=1, rho resonances (mode 1,2,3)

isoVectorMasses = [775.26*par.MeV,1.74, 1.8975485530633405]
isoVectorWidths = [149.1 *par.MeV, 0.35, 0.5037543149729891]
isoVectorAmp   = [0. , 0. , 13.505485155732044 ]
isoVectorPhase = [0., 0., 0.]

I1_phase3 = 0.0
####################################################################################

#
###############################
## Parameter set for DM part ##
###############################
#
gDM_ = 1.
mDM_ = 0.41
mMed_ = 5
wMed_ = 10.
cI1_ = 1.
cI0_ = 1.
cS_ = 1.
###############################


###############################
# function to reset parameters#
###############################
def resetParameters(gDM,mDM,mMed,wMed,cMedu,cMedd,cMeds) :
    global gDM_,mDM_,mMed_,wMed_, cI1_, cI0_,cS_
    cI1_ = cMedu-cMedd
    # cI0_ = 3*(cMedu+cMedd)
    cS_ = -3*cMeds
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed


#
#############################
## form factor calculation ##
#############################
#
def pcm(m02,m12,m22) :
    return 0.5*math.sqrt((m02**2+m12**2+m22**2-2.*m02*m12-2.*m02*m22-2.*m12*m22)/m02)

# different modes:
# mode==0: K*(890)K-pi+ + cc. -> K+pi-K-pi+
# mode==1: K*0(890)K+-pi-+  -> KS K+- pi-+ pi0
# mode==2: K*+-(890)KSpi-+  -> KS K+- pi-+ pi0
# mode==3: K*+-(890)K-+pi0  -> KS K+- pi-+ pi0

# Integrand for m_{K\pi}^2 integration, integrate over unknown K\pi structure, see eq. (62) low_energy.pdf

def integrand_Kpi(mKpi,s,mode) :
    # mK+-, mPi-+
    if mode==0:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK+-, mPi-+
    if mode==1:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK0, mPi-+
    if mode==2:
        mPi = mPi_[1]
        mK = mK_[1]
    # mK-+, mPi0
    if mode==3:
        mPi = mPi_[0]
        mK = mK_[0]
    output=[]
    for val in mKpi :
        Q2 = val
        Q = math.sqrt(Q2)
        P2 = pcm(Q2,mK**2,mPi**2)
        P3 = pcm(s,Q2,mKstar_**2)
        mom_term = P2*P3/Q*(1+P3**2/3./mKstar_**2)
        output.append(mom_term)
    return  output

# Integration over m_{K\pi}i^2
def phase(s,mode) :
    # mK+-, mPi-+
    if mode==0:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK+-, mPi-+
    if mode==1:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK0, mPi-+
    if mode==2:
        mPi = mPi_[1]
        mK = mK_[1]
    # mK-+, mPi0
    if mode==3:
        mPi = mPi_[0]
        mK = mK_[0]
    if s<(mKstar_+mPi+mK)**2:
        return 0
    upp = (math.sqrt(s)-mKstar_)**2
    low = (mK+mPi)**2
    return scipy.integrate.quadrature(integrand_Kpi,low,upp,args=(s,mode),tol=1e-10,maxiter=200)[0]

#form factor
def FKstarKpi(s,mode):
    form = 0.
    AI0_0 = 0.
    AI1_0 = 0.
    AI0_ = 0.
    AI1_ = 0.
    if mode==0:
        # I=0
        for i in range(0,len(isoScalarMasses0)):
            AI0_0+=cS_*isoScalarAmp0[i]*Resonance.BreitWignerFW(s,isoScalarMasses0[i],isoScalarWidths0[i])*cmath.exp(ii*isoScalarPhase0[i])
        # I=1
        for i in range(0,len(isoVectorMasses0)):
            AI1_0+=cI1_*isoVectorAmp0[i]*Resonance.BreitWignerFW(s,isoVectorMasses0[i],isoVectorWidths0[i])*cmath.exp(ii*isoVectorPhase0[i])
        form=math.sqrt(2/9.)*(AI0_0+AI1_0) #K*0 -> K+\pi-
    else:
         # I=0
        for i in range(0,len(isoScalarMasses)):
            AI0_+=cS_*isoScalarAmp[i]*Resonance.BreitWignerFW(s,isoScalarMasses[i],isoScalarWidths[i])*cmath.exp(ii*isoScalarPhase[i])
        # I=1
        for i in range(0,len(isoVectorMasses)):
            AI1_+=cI1_*isoVectorAmp[i]*Resonance.BreitWignerFW(s,isoVectorMasses[i],isoVectorWidths[i])*cmath.exp(ii*isoVectorPhase[i]) 
        if mode==1:
            form+=math.sqrt(1/18.)*(AI1_+AI0_) #K*0 -> K0\pi0
        if mode==2:
            form=math.sqrt(1./18.)*(AI1_- AI0_) #K* pm -> Kpm \pi0
        if mode==3:
            form=math.sqrt(1./18.)*(AI1_- AI0_) #K* pm ->  K0 \pi pm
    return form

#
###############
## processes ##
###############
#

# cross-section for e+e- -> KKpipi for certain mode
def sigmaSM_mode(s,mode) :
    # mK+-, mPi-+
    if mode==0:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK+-, mPi-+
    if mode==1:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK0, mPi-+
    if mode==2:
        mPi = mPi_[1]
        mK = mK_[1]
    # mK-+, mPi0
    if mode==3:
        mPi = mPi_[0]
        mK = mK_[0]
    if s<(mKstar_+mPi+mK)**2:
        return 0
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s # SM
    pre *= 3/64/math.pi**3/s**1.5 # phase-space
    return pre*phase(s,mode)*par.gev2nb*abs(FKstarKpi(s,mode))**2

# cross-section for e+e- -> KKpipi for certain mode
def sigmaSM(s):
    sigmatot = 0.
    for imode in [0,1,2,3]:
        sigmatot += sigmaSM_mode(s,imode)
    return sigmatot

# Dark

# Decay rate of mediator-> KKpipi for certain mode
def GammaDM_mode(mMed,mode) :
    # mK+-, mPi-+
    if mode==0:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK+-, mPi-+
    if mode==1:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK0, mPi-+
    if mode==2:
        mPi = mPi_[1]
        mK = mK_[1]
    # mK-+, mPi0
    if mode==3:
        mPi = mPi_[0]
        mK = mK_[0]
    if mMed**2<(mKstar_+mPi+mK)**2:
        return 0
    # vector spin average
    pre = 1/3.
    # phase-space
    pre *= (3/64/math.pi**3/mMed**3)*mMed
    return pre*abs(phase(mMed**2,mode))*abs(FKstarKpi(mMed**2,mode))**2

# Decay rate of mediator-> KKpipi total
def GammaDM(mMed):
    Gammatot = 0
    for i in range(0,4):
        Gammatot+=GammaDM_mode(mMed,i)
    return Gammatot

# cross-section for e+e- -> KKpipi for certain mode
def sigmaDM_mode(s,mode) :
    # mK+-, mPi-+
    if mode==0:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK+-, mPi-+
    if mode==1:
        mPi = mPi_[1]
        mK = mK_[0]
    # mK0, mPi-+
    if mode==2:
        mPi = mPi_[1]
        mK = mK_[1]
    # mK-+, mPi0
    if mode==3:
        mPi = mPi_[0]
        mK = mK_[0]
    if s<(mKstar_+mPi+mK)**2:
        return 0
    # Dark
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre= DMmed2*s*(1+2*mDM_**2/s)/3.
    # phase-space
    pre *= 3/64/math.pi**3/s**1.5
    return pre*phase(s,mode)*par.gev2nb*abs(FKstarKpi(s,mode))**2

# cross-section for e+e- -> KKpipi total
def sigmaDM(s):
    sigmatot = 0.
    for imode in range(0,4):
        sigmatot += sigmaDM_mode(s,imode)
    return sigmatot