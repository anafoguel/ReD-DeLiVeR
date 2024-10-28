# Libraries to load
import scipy.integrate
import math,cmath
import os
import matplotlib.pyplot as plt
from src.functions import alpha,Resonance
import src.pars as par

# define complex number
ii = complex(0.,1.)

#
####################################################################################
## Parameter set for hadronic current, own parametrization                        ##
####################################################################################
#

mPi_ = [0.1349770,0.13957018] # mass [Pi0, Pi+-] GeV
mK_  = [0.493677,0.497611] # mass [K+-, K0] GeV
m_F0_ = 0.975
g_F0_ = 0.1
a_F0_ = 1.
phasef0_ = 0.
wPiPi_ = 0.196 # GammaPiPi GeV
gKK_ = 2.51 
gPiPi_ = 0.417
g12_       = .165
g22_       = .695
# Phi Resonances 
mPhi_ = [1.019461,1.680206376111421,1.85,2.162094475680359] # masses PDG
# gPhi_ = [ 0.004249, 0.218,  0.087] # widths PDG
gPhi_ = [ 0.004249, 0.22649014716417465,  0.1,0.20905046718427628]
aPhi_ = [0.,1.4440212871376037,0.,0.6856322620088697] # contribution only of phi(1680)
phasePhi_ = [0.,2.62513009401222,0.,0.]
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
    cI0_ = 3*(cMedu+cMedd)
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
def BWs(a,s, mMed, gMed) :
    return a*mMed**2/((mMed**2-s)-ii*math.sqrt(s)*gMed)

def BW(a,s, mMed, gMed) :
    return a*mMed**2/((mMed**2-s)-ii*mMed*gMed)

def pcm(m02,m12,m22) :
    return 0.5*math.sqrt((m02**2+m12**2+m22**2-2.*m02*m12-2.*m02*m22-2.*m12*m22)/m02)


def integrand(mpp,s,mode) :
    output=[]
    sqrts = math.sqrt(s)
    for val in mpp :
        Q2 = val#m_F0_*g_F0_*math.tan(val)+m_F0_**2
        Q = math.sqrt(Q2)
        pre = 1.
        P2 = pcm(Q2,mPi_[mode]**2,mPi_[mode]**2)
        P3 = pcm(s,Q2,mPhi_[0]**2)
        mom_term = P2*P3/Q*(1+P3**2/3./mPhi_[0]**2)
        output.append(pre*mom_term)
    return  output

        
# Integration over mpipi^2 in omega f0 channel
def phase(s,mode) :
    if s<(mPhi_[0]+2*mPi_[mode])**2:
        return 0
    upp = (math.sqrt(s)-mPhi_[0])**2
    low = 4.*mPi_[mode]**2
    pre = 1
    if mode==0: pre /=2.
    return pre*scipy.integrate.quadrature(integrand,low,upp,args=(s,mode),tol=1e-10,maxiter=200)[0]

#
###############
## processes ##
###############
#

# cross-section for e+e- -> Phi Pi Pi
def sigmaSM_mode(s,mode) :
    if s<(mPhi_[0]+2*mPi_[mode])**2:
        return 0
    med_prop = 0.
    for i in range(0,len(mPhi_)):
        med_prop+=cS_*BWs(aPhi_[i],s,mPhi_[i],gPhi_[i])*cmath.exp(ii*phasePhi_[i])
    sigma =0
    sigma += phase(s,mode)*par.gev2nb*abs(med_prop)**2
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s # SM
    pre *= 3/64/math.pi**3/s**1.5 # phase-space
    return pre*sigma

# cross-section for e+e- -> Phi Pi Pi
def sigmaSM(s):
    sigmatot = 0.
    for imode in [0,1]:
        sigmatot += sigmaSM_mode(s,imode)
    return sigmatot


########### Dark ############

# Decay rate of mediator-> Phi Pi Pi to certain mode
def GammaDM_mode(mMed,mode) :
    if mMed**2<(mPhi_[0]+2*mPi_[mode])**2:
        return 0
    # vector spin average
    pre = 1/3.
    # phase-space
    pre *= (3/64/math.pi**3/mMed**3)*mMed
    med_prop = 0.
    for i in range(0,len(mPhi_)):
        med_prop+=cS_*BWs(aPhi_[i],mMed**2,mPhi_[i],gPhi_[i])*cmath.exp(ii*phasePhi_[i])
    sigma =0
    sigma += phase(mMed**2,mode)*abs(med_prop)**2
    return pre*abs(sigma)

# Decay rate of mediator-> Phi Pi Pi
def GammaDM(mMed):
    Gammatot = 0
    for i in range(0,2):
        Gammatot+=GammaDM_mode(mMed,i)
    return Gammatot

# cross section for DM annihilations
def sigmaDM(s,mode) :
    if s<(mPhi_[0]+2*mPi_[mode])**2:
        return 0
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre= DMmed2*s*(1+2*mDM_**2/s)/3.
    #coming from phase space 
    pre *= 3/64./math.pi**3/s**1.5
    med_prop = 0.
    for i in range(0,len(mPhi_)):
        med_prop+=cS_*BWs(aPhi_[i],s,mPhi_[i],gPhi_[i])*cmath.exp(ii*phasePhi_[i])
    sigma =0
    sigma += phase(s,mode)*par.gev2nb*abs(med_prop)**2
    return pre*sigma

