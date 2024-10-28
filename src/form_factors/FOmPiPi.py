# Libraries to load
import math,scipy.integrate,cmath
import os
import matplotlib.pyplot as plt
from src.functions import alpha,Resonance
import src.pars as par

#define complex number
ii = complex(0.,1.)

#
####################################################################################
## Parameter set for hadronic current, own parametrization                        ##
####################################################################################
#
cF0_       = [.165,.695]
# PDG values
mPi_       = [0.1349770,0.13957018]
mK_        = [0.493677,0.497611]

# own parametrization, see arXiv:1911.11147

m_F0_       = 0.990
g_F0_       = 0.1
mF0_       = [0.600,0.980]
gF0_       = [1.0,0.10]
aF0_ = [1.,0.883]
phasef0_ = 0.
#mB1_       = 1.2295
#gB1_       = 0.142
mOm_ = 0.78265 #omega mass for decay product
mOmega_     = [ 0.783,1.420, 1.6608543573197]# omega masses for vector meson mixing
gOmega_     = [0.00849 ,0.315,0.3982595005228462]
aOm_ = [0.,0.,2.728870588760009]
phaseOm_ = [0.,math.pi,0.]
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
    if ((m02**2+m12**2+m22**2-2.*m02*m12-2.*m02*m22-2.*m12*m22)/m02)<0:
        print ((m02**2+m12**2+m22**2-2.*m02*m12-2.*m02*m22-2.*m12*m22)/m02,m02,m12,m22 )
    return 0.5*math.sqrt((m02**2+m12**2+m22**2-2.*m02*m12-2.*m02*m22-2.*m12*m22)/m02)


def production_vec(s,mMed, gMed,a_s,phaseOm):
    pre = alpha.alphaEM(s)**2/48./math.pi/s**3
    med_prop=0.
    for i in range(0,len(mMed)):
        med_prop+=cI0_*BWs(a_s[i],s,mMed[i],gMed[i])*cmath.exp(ii*phaseOm[i])
    return pre*abs(med_prop)**2



# Integrand for mpipi^2 integration
def integrand_f0(mpp,s,mode) :
    output=[]
    sqrts = math.sqrt(s)
    for val in mpp :
        Q2 = val#m_F0_*g_F0_*math.tan(val)+m_F0_**2
        Q = math.sqrt(Q2)
        pre = 1.
        #pre *= ((Q2-m_F0_**2)**2 + (m_F0_*g_F0_)**2)/m_F0_/g_F0_
        form_f0 = 1.
        #form_f0 *= f0_flatte(Q2,mF0_,gF0_,aF0_,phasef0_)
        P2 = pcm(Q2,mPi_[mode]**2,mPi_[mode]**2)
        P3 = pcm(s,Q2,mOm_**2)
        mom_term = P2*P3/Q*(1+P3**2/3./mOm_**2)
        output.append(pre*mom_term*form_f0)
    return  output

        
# Integration over mpipi^2 in omega f0 channel
def phase(s,mode) :
    if s<(mOm_+2*mPi_[mode])**2:
        return 0
    upp = (math.sqrt(s)-mOm_)**2
    low = 4.*mPi_[1]**2
    #upp = math.atan((upp-m_F0_**2)/g_F0_/m_F0_)
    #low = math.atan((low-m_F0_**2)/g_F0_/m_F0_)
    pre = 1.
    if mode==0: pre /=2.
    return pre*scipy.integrate.quadrature(integrand_f0,low,upp,args=(s,mode),tol=1e-10,maxiter=200)[0]

#
###############
## processes ##
###############
#

# cross-section for e+e- -> Omega Pi Pi to certain mode
def sigmaSM_mode(s,mode) :
    if s<=(mOm_+2*mPi_[1])**2:
        return 0
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    pre *= 3/64./math.pi**3/s**1.5
    med_prop = 0.
    for i in range(0,len(mOmega_)):
        med_prop+=cI0_*BWs(aOm_[i],s,mOmega_[i],gOmega_[i])*cmath.exp(ii*phaseOm_[i])
    return pre*abs(phase(s,mode))*par.gev2nb*abs(med_prop)**2

# cross-section for e+e- -> Omega Pi Pi total
def sigmaSM(s):
    sigmatot = 0.
    for imode in [0,1]:
        sigmatot += sigmaSM_mode(s,imode)
    return sigmatot


# Dark
# Decay rate of mediator-> Omega PiPi to certain mode
def GammaDM_mode(mMed, mode) :
    if mMed**2<=(2*mPi_[1]+mOm_)**2: return 0
    if cI0_==0: return 0
    # vector spin average
    pre = 1/3.
    #coming from phase space 
    pre *= 3/64./math.pi**3/mMed**2
    med_prop = 0.
    for i in range(0,len(mOmega_)):
        med_prop+=cI0_*BWs(aOm_[i],mMed**2,mOmega_[i],gOmega_[i])*cmath.exp(ii*phaseOm_[i])
    return pre*phase(mMed**2,mode)*abs(med_prop)**2

# Decay rate of mediator-> Omega PiPi total
def GammaDM(mMed):
    Gammatot = 0
    for i in range(0,2):
        Gammatot+=GammaDM_mode(mMed,i)
    return Gammatot


# cross section for DM annihilations to certain mode
def sigmaDM_mode(s, mode):
    if s<(2*mPi_[mode]+mOm_)**2: return 0
    if cI0_==0: return 0
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre= DMmed2*s*(1+2*mDM_**2/s)/3.
    #coming from phase space 
    pre *= 3/64./math.pi**3/s**1.5
    med_prop = 0.
    for i in range(0,len(mOmega_)):
        med_prop+=cI0_*BWs(aOm_[i],s,mOmega_[i],gOmega_[i])*cmath.exp(ii*phaseOm_[i])
    return pre*phase(s,mode)*par.gev2nb*abs(med_prop)**2

# cross section for DM annihilations
def sigmaDM(s):
    sigDM = 0.
    for i in range(0,2):
        sigDM+=sigmaDM_mode(s,i)
    return sigDM

