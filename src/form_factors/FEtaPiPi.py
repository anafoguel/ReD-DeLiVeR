# Libraries to load
import math,scipy.integrate,cmath
import os
import matplotlib.pyplot as plt
import random
import statistics
from src.functions import alpha,Resonance
import src.pars as par

#define complex number
ii = complex(0.,1.)


#
####################################################################################
## Parameter set for hadronic current, parametrization taken from arXiv:1306.1985 ##
######################## with fit from arXiv:1911.11147 ############################
####################################################################################
#
mEta_     = 0.547862
mPi_      = 0.13957018 
fPi_      = 0.0933
mRho_     = [0.77549,1.54,1.76,2.15]
gRho_     = [0.1494 ,0.356,0.113 ,0.32 ]
amp_      = [1.,0.326,0.0115,0.]
phase_    = [0,3.14,3.14,0.]
camp_ = []
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
    global gDM_,mDM_,mMed_,wMed_, cI1_, cI0_, cS_
    cI1_ = cMedu-cMedd
    cI0_ = 3*(cMedu+cMedd)
    cS_ = -3*cMeds
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    
for i in range(0,len(amp_)):
    camp_.append(amp_[i]*cmath.exp(ii*phase_[i]))# [1.,a1*cmath.exp(ii*phi1),a2*cmath.exp(ii*phi2),a3*cmath.exp(ii*phi3)]

#
#############################
## form factor calculation ##
#############################
#

# rho width within the propagator
def gammaRho(i,Q2) :
    if i==0 :
        return gRho_[0]*mRho_[0]**2/Q2*\
            ((Q2-4.*mPi_**2)/(mRho_[0]**2-4.*mPi_**2))**1.5
    else :
        return gRho_[i]*Q2/mRho_[i]**2

# Breit-Wigner functions, can probably also be found in Resonance.py
def BW(i,Q2) :
    gam = gammaRho(i,Q2)
    mr2 = mRho_[i]**2
    return mr2/(mr2-Q2-ii*math.sqrt(Q2)*gam)
    
# momentum as in eq. (48.17) in https://pdg.lbl.gov/2020/reviews/rpp2020-rev-kinematics.pdf 
def pcm(m02,m12,m22) :
    return 0.5*math.sqrt((m02**2+m12**2+m22**2-
                          2.*m02*m12-2.*m02*m22-2.*m12*m22)/m02)

#Form-factor of EtaPiPi, s is the c.m. energy squared of the process,
# and Q2 is the four-momentum squared of the intermediate rho state
def FEtaPiPi(s,Q2):
    total = sum(camp_)
    for i in range(0,len(camp_)) :
        camp_[i] /=total
    form = 0.
    pre = 0.25/math.sqrt(3.)/math.pi**2/fPi_**3
    for i in range(0,len(camp_)) :
        form += BW(i,s)*camp_[i]*cI1_
    form*=BW(0,Q2)
    return pre*form

# integrand for phase space integration
def integrand(rho,s) :
    output=[]
    for val in rho :
        Q2 = mRho_[0]*gRho_[0]*math.tan(val)+mRho_[0]**2
        peta = pcm(s,mEta_**2,Q2)
        ppi  = pcm(Q2,mPi_**2,mPi_**2)
        Q    = math.sqrt(Q2)
        # due to change of variables in integration
        pre = ((Q2-mRho_[0]**2)**2 + (mRho_[0]*gRho_[0])**2)/mRho_[0]/gRho_[0]
        output.append(pre*(peta*ppi)**3/Q*abs(FEtaPiPi(s,Q2))**2)
    return output

# phase space integration
def phase(s) :
    upp = (math.sqrt(s)-mEta_)**2
    low = 4.*mPi_**2
    upp = math.atan((upp-mRho_[0]**2)/gRho_[0]/mRho_[0])
    low = math.atan((low-mRho_[0]**2)/gRho_[0]/mRho_[0])
    return scipy.integrate.quadrature(integrand,low,upp,args=s,tol=1e-12,maxiter=200)[0]

#
###############
## processes ##
###############
#

# Decay rate of dark mediator -> Eta Pi Pi
def GammaDM(mMed) :
    if mMed**2<(2*mPi_+mEta_)**2: return 0
    if cI1_==0: return 0
    # vector spin average
    pre = 1/3.
    #coming from phase space 
    pre*=(1./12./(2.*math.pi)**3/mMed)*mMed
    return pre*abs(phase(mMed**2))

# DM cross section for Eta Pi Pi
def sigmaDM(s):
    if s<(2*mPi_+mEta_)**2: return 0
    if cI1_==0: return 0
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre= DMmed2*s*(1+2*mDM_**2/s)/3.
    #coming from phase space 
    pre*=1./12./(2*math.pi)**3/s**0.5 
    return pre*phase(s)*par.gev2nb
    
# cross-section for e+e- -> Eta Pi Pi
def sigmaSM(s) :
    if s<(2*mPi_+mEta_)**2: return 0
    if cI1_==0: return 0
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    #coming from phase space 
    pre*=1./12./(2*math.pi)**3/s**0.5 
    return pre*phase(s)*par.gev2nb
