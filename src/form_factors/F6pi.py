# Libaries to load
import numpy,scipy,math,cmath,os
import matplotlib.pyplot as plt
from src.functions import alpha,Resonance
import src.pars as par

# define complex number
ii= complex(0,1)

# In the following we label the channels with c for fully charged pion channel 3(pi+pi-) and n for the channel including two neutral pions 2(pi+pi-)2pi0
#
###########################################################################
## Parametrization taken from https://arxiv.org/pdf/hep-ex/0602006v1.pdf ##
###########################################################################
#
# Resonance values -> given in paper hep-ex/0602006
# In the following we label the channels with c for fully charged pion channel 3(pi+pi-) and n for the channel including two neutral pions 2(pi+pi-)2pi0

mPi_       = [0.1349770,0.13957018]

# 3(pi+pi-) 
c_c_ = 0.0036595657695477346
m_c_ = 1.88
phi_c_ = math.radians(21.)
Gamma_c_ = 0.13

c0_c_ = 0.015278568501613186
c1_c_ = -1.0819476867207243
a_c_ = 0.8896494373262481
b_c_ = 1.4
m0_c_ = 1.26158551198849

# 2(pi+pi-)2pi0
c_n_ = -0.0072157182261625066
m_n_ = 1.86
phi_n_ = - math.radians(3.)
Gamma_n_ = 0.16

c0_n_ = -0.027580774291809617
c1_n_ = 2.3792205359064202
a_n_ = 0.8543454953801702
b_n_ = 1.5431907628236239
m0_n_ = 1.19907252783525

#
###############################
## Parameter set for DM part ##
###############################
#
cI1_ = 1.
cI0_ = 0.
cS_ = 0.


#change rho, omega, phi contributions
def resetParameters(gDM,mDM,mMed,wMed,cMedu,cMedd,cMeds) :
    global gDM_,mDM_,mMed_,wMed_
    global cI1_,cI0_,cS_
    cI1_ = cMedu-cMedd
    # cI0_ = 3*(cMedu+cMedd)
    # cS_ = -3*cMeds
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    
#
###############
## processes ##
###############
#   
    
def sigmaSM_mode(s,mode):
    if s<(1.5)**2: return 0
    sqrts= math.sqrt(s)
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    if mode==0:
        pre *= 3./4/math.pi/math.sqrt(s)
        form = c_n_*m_n_**2*cmath.exp(ii*phi_n_)/(s-m_n_**2+ii*sqrts*Gamma_n_)
        cont=c0_n_
        cont+=c1_n_*(cmath.exp(-b_n_/(sqrts-m0_n_))/(sqrts-m0_n_)**(2-a_n_))
        form+=cont
        form*=cI1_
    if mode==1:
        pre *= 3./4/math.pi/math.sqrt(s)
        form =c_c_*m_c_**2*cmath.exp(ii*phi_c_)/(s-m_c_**2+ii*sqrts*Gamma_c_)
        cont=c0_c_
        cont+=c1_c_*(cmath.exp(-b_c_/(sqrts-m0_c_))/(sqrts-m0_c_)**(2-a_c_))
        form+=cont
        form*=cI1_
    return pre*abs(form)**2*par.gev2nb
    
def sigmaSM(s):
    sigmatot = 0.
    sigmatot += sigmaSM_mode(s,0)
    sigmatot += sigmaSM_mode(s,1)
    return sigmatot

# Dark 
def GammaDM_mode(mMed,mode):
    if mMed**2<=(1.4)**2: return 0
    if cI1_==0: return 0
    # vector spin average
    pre = 1/3.
    # parametrization 
    pre *=  (3./4/math.pi/mMed)*mMed
    form = 0
    if mode==0:
        # parametrization 
        form = c_n_*m_n_**2*cmath.exp(ii*phi_n_)/(mMed**2-m_n_**2+ii*mMed*Gamma_n_)
        cont=c0_n_
        cont+=c1_n_*(cmath.exp(-b_n_/(mMed-m0_n_))/(mMed-m0_n_)**(2-a_n_))
        form+=cont
        form*=cI1_
    if mode==1:
        # parametrization 
        form =c_c_*m_c_**2*cmath.exp(ii*phi_c_)/(mMed**2-m_c_**2+ii*mMed*Gamma_c_)
        cont=c0_c_
        cont+=c1_c_*(cmath.exp(-b_c_/(mMed-m0_c_))/(mMed-m0_c_)**(2-a_c_))
        form+=cont
        form*=cI1_
    return pre*abs(form)**2
        
def GammaDM(mMed):
    Gammatot=0
    Gammatot+=GammaDM_mode(mMed,0)
    Gammatot+=GammaDM_mode(mMed,1)
    return Gammatot


def sigmaDM_mode(s,mode):
    if cI1_==0: return 0
    sqrts= math.sqrt(s)
    # Dark
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre= DMmed2*s*(1+2*mDM_**2/s)/3.
    # parametrization
    if mode==0:
        pre *= 3./4/math.pi/sqrts
        form = c_n_*m_n_**2*cmath.exp(ii*phi_n_)/(s-m_n_**2+ii*sqrts*Gamma_n_)
        cont=c0_n_
        cont+=c1_n_*(cmath.exp(-b_n_/(sqrts-m0_n_))/(sqrts-m0_n_)**(2-a_n_))
        form+=cont
        form*=cI1_
    if mode==1:
        pre *= 3./4/math.pi/sqrts
        form =c_c_*m_c_**2*cmath.exp(ii*phi_c_)/(s-m_c_**2+ii*sqrts*Gamma_c_)
        cont=c0_c_
        cont+=c1_c_*(cmath.exp(-b_c_/(sqrts-m0_c_))/(sqrts-m0_c_)**(2-a_c_))
        form+=cont
        form*=cI1_
    return pre*abs(form)**2*par.gev2nb

def sigmaDM(s):
    sigDM = sigmaDM_mode(s,0)
    sigDM += sigmaDM_mode(s,1)
    return sigDM




