# Libraries to load
import numpy,scipy,math,cmath,os
import matplotlib.pyplot as plt
from src.functions import alpha,Resonance
import src.pars as par

# define complex number 
ii = complex(0.,1.)

#
####################################################################################
## Parameter set for hadronic current, parametrization taken from arXiv:1407.7995 ##
## with fit values from 1911.11147                                                ##
####################################################################################
#
# masses and widths of rho resonances                                                              
rhoMasses_   = [0.77549, 1.465, 1.720, 2.12,2.32647];
rhoWidths_   = [0.14910,  0.400,  0.250, 0.3 ,0.4473 ];
# masses and width of omega resonances                                                             
omegaMasses_ = [0.78265, 1.425, 1.670, 2.0707 , 2.34795 ];
omegaWidths_ = [0.00849  ,  0.215,  0.315, 1.03535, 1.173975];
# c_1 couplings, real and imaginary part                                                                                    
c1Re_ = [1.,-0.467,-0.177,0.301]
c1Im_ = [0.,-0.385,0.149, 0.264]
# c_2 couplings, real and imaginary part                                                                                    
c2Re_ = [1.,0.0521,-0.00308,-0.348]
c2Im_ = [0.,-3.04,2.38,-0.104]
# c_3 couplings, real and imaginary part                                                           
c3Re_ = [1.,-7.88,10.2]
c3Im_ = [0.,5.67,-1.94]
# c_4 couplings, real and imaginary part                                                                                    
c4Re_ = [1.,-0.832,0.405]
c4Im_ = [0.,0.308,-0.25]
# Magnetic moments                                                                                 
mup_ =  2.793;
mun_ = -1.913;

# parameter a and b
a = mup_-mun_-1
b = -mup_-mun_+1

# proton mass
mp_ = 0.9382720813
mn_ = 0.9395654133

c1_ = []
c2_ = []
c3_ = []
c4_ = []
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
    global c1_, c2_, c3_,c4_
    cI1_ = cMedu-cMedd
    cI0_ = 3*(cMedu+cMedd)
    cS_ = -3*cMeds
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    c1_ = []
    c2_ = []
    c3_ = []
    c4_ = []
    initialize()
    
################################
# initialize resonances        #
################################
def initialize() :
    global c1_,c2_,c3_,c4_
    # calculate c1i couplings and append it to coupling
    fact = 0.
    for i in range(0,len(c1Re_)):
        c1_.append(c1Re_[i]+ii*c1Im_[i])
        fact+=c1_[i]*omegaMasses_[i]**2
    c1_.append(-fact/omegaMasses_[4]**2)
    
    # calculate c2i couplings and append it to coupling
    fact = 0.
    for i in range(0,len(c2Re_)):
        c2_.append(c2Re_[i]+ii*c2Im_[i])
        fact+=c2_[i]*rhoMasses_[i]**2
    c2_.append(-fact/rhoMasses_[4]**2)
    
    # calculate c3i couplings and append it to coupling
    fact=0.
    fact2=0.
    for i in range(0,3):
        c3_.append(c3Re_[i]+ii*c3Im_[i])
        fact+= c3_[i]*omegaMasses_[i]**2
        fact2+= c3_[i]*omegaMasses_[i]**2*(omegaMasses_[i]**2-omegaMasses_[4]**2
                                        +ii*(omegaMasses_[4]*omegaWidths_[4]-omegaMasses_[i]*omegaWidths_[i]))
    c3_.append(fact2/omegaMasses_[3]**2/
            (omegaMasses_[4]**2-omegaMasses_[3]**2
                +ii*(omegaMasses_[3]*omegaWidths_[3]-omegaMasses_[4]*omegaWidths_[4])))
    fact +=c3_[3]*omegaMasses_[3]**2
    c3_.append(-fact/omegaMasses_[4]**2)

    # calculate c4i couplings and append it to coupling
    fact=0.
    fact2=0.
    for i in range(0,3):
        c4_.append(c4Re_[i]+ii*c4Im_[i])
        fact+= c4_[i]*rhoMasses_[i]**2
        fact2+= c4_[i]*rhoMasses_[i]**2*(rhoMasses_[i]**2-rhoMasses_[4]**2
                                        +ii*(rhoMasses_[4]*rhoWidths_[4]-rhoMasses_[i]*rhoWidths_[i]))
    c4_.append(fact2/rhoMasses_[3]**2/
            (rhoMasses_[4]**2-rhoMasses_[3]**2
                +ii*(rhoMasses_[3]*rhoWidths_[3]-rhoMasses_[4]*rhoWidths_[4])))
    fact +=c4_[3]*rhoMasses_[3]**2
    c4_.append(-fact/rhoMasses_[4]**2)

#
#############################
## form factor calculation ##
#############################
#
# Breit Wigner width for Form factors
def BW(s,m,Gamma):
    return m**2/(m**2-s-ii*Gamma*m)


# Form factor components
def F1s(s):
    F1S = 0.
    n1 = 0.
    #print "size of c1 ", len(c1_)
    for i in range(0,len(c1_)):
        F1S += c1_[i]*BW(s,omegaMasses_[i],omegaWidths_[i])
        n1 +=c1_[i]
    F1S *=0.5/n1
    return cI0_*F1S

def F1v(s):
    F1V = 0.
    n2 = 0.
    for i in range(0,len(c2_)):
        F1V += c2_[i]*BW(s,rhoMasses_[i],rhoWidths_[i])
        n2 +=c2_[i]
    F1V *=0.5/n2
    return cI1_*F1V

def F2s(s):
    F2S = 0.
    n3 = 0.
    for i in range(0,len(c3_)):
        F2S += c3_[i]*BW(s,omegaMasses_[i],omegaWidths_[i])
        n3+=c3_[i]
    F2S *=-0.5*b/n3
    return cI0_*F2S

def F2v(s):
    F2V = 0.
    n4 = 0.
    for i in range(0,len(c4_)):
        F2V += c4_[i]*BW(s,rhoMasses_[i],rhoWidths_[i])
        n4 +=c4_[i]
    F2V *= 0.5*a/n4
    return cI1_*F2V

# Proton form factors
def F1P(s):
    return F1s(s)+F1v(s)

def F2P(s):
    return F2s(s)+F2v(s)

#Neutron form factors
def F1N(s):
    return F1s(s)-F1v(s)

def F2N(s):
    return F2s(s)-F2v(s)


#
###############
## processes ##
###############
#

# Based on eq. (12) in hep-ph/0403062
# cross-section for e+e- -> p pbar
def sigmaP(s):
    if s<4*mp_**2:
        return 0
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    pre *= 3.*(1-4.*mp_**2/s)**0.5/64./math.pi**2
    pre *= 2.*math.pi #from integration over phi angle
    tau = s/4./mp_**2
    GM = F1P(s)+F2P(s)
    GE = F1P(s)+tau*F2P(s)
    #print "form factors ", GM, GE
    form =8./3*abs(GM)**2+4./3/tau*abs(GE)**2 #factors of 8/3, 4/3 coming from integration over theta
    return pre*form*par.gev2nb

# cross-section for e+e- -> n nbar
def sigmaN(s):
    if s<4*mn_**2:
        return 0
    mode="timelike"
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    pre *= 3.*(1-4.*mn_**2/s)**0.5/64./math.pi**2
    pre*=2.*math.pi #from integration over phi angle
    tau = s/4./mn_**2
    #c1,c2,c3,c4 = initialize_tl(c1Re_,c1Im_,c2Re_,c2Im_,c3Re_,c3Im_,c4Re_,c4Im_)
    GM = F1N(s)+F2N(s)
    GE = F1N(s)+tau*F2N(s)
    form =8./3.*abs(GM)**2+4./3./tau*abs(GE)**2 #factors of 8/3, 4/3 coming from integration over theta
    return pre*form*par.gev2nb

#Dark
# Decay rate of mediator-> p pbar
def GammaPDM(mMed) :
    if mMed**2<4*mp_**2: return 0
    # vector spin average
    pre = 1/3.
    #coming from phase space 
    pre *= 3.*mMed*(1-4.*mp_**2/mMed**2)**0.5/64./math.pi**2
    pre *= 2.*math.pi #from integration over phi angle
    tau = mMed**2/4./mp_**2
    GM = F1P(mMed**2)+F2P(mMed**2)
    GE = F1P(mMed**2)+tau*F2P(mMed**2)
    #print "form factors ", GM, GE
    form =8./3*abs(GM)**2+4./3/tau*abs(GE)**2 #factors of 8/3, 4/3 coming from integration over theta
    return pre*form

# Decay rate of mediator-> n nbar
def GammaNDM(mMed) :
    if mMed**2<4*mn_**2: return 0
    # mode="timelike"
    # vector spin average
    pre = 1/3.
    #coming from phase space 
    pre *= 3.*mMed*(1-4.*mn_**2/mMed**2)**0.5/64./math.pi**2
    pre *= 2.*math.pi #from integration over phi angle
    tau = mMed**2/4./mn_**2
    #c1,c2,c3,c4 = initialize_tl(c1Re_,c1Im_,c2Re_,c2Im_,c3Re_,c3Im_,c4Re_,c4Im_)
    GM = F1N(mMed**2)+F2N(mMed**2)
    GE = F1N(mMed**2)+tau*F2N(mMed**2)
    #print "form factors ", GM, GE
    form =8./3*abs(GM)**2+4./3/tau*abs(GE)**2 #factors of 8/3, 4/3 coming from integration over theta
    return pre*form

def GammaDM_mode(mMed,mode):
    Gamma = 0
    if mode==0:
        Gamma = GammaPDM(mMed)
    if mode==1:
        Gamma = GammaNDM(mMed)
    return Gamma
    
# cross section for DM annihilations to protons
def sigmaPDM(s):
    if s<4*mp_**2:
        return 0
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre= DMmed2*s*(1+2*mDM_**2/s)/3.
    pre *= 3.*(1-4.*mp_**2/s)**0.5/64./math.pi**2
    pre *= 2.*math.pi #from integration over phi angle
    tau = s/4./mp_**2
    GM = F1P(s)+F2P(s)
    GE = F1P(s)+tau*F2P(s)
    #print "form factors ", GM, GE
    form =8./3*abs(GM)**2+4./3/tau*abs(GE)**2 #factors of 8/3, 4/3 coming from integration over theta
    return pre*form*par.gev2nb

# cross section for DM annihilations to neutrons
def sigmaNDM(s):
    if s<4*mn_**2:
        return 0
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre= DMmed2*s*(1+2*mDM_**2/s)/3.
    pre *= alpha.alphaEM(s)**2*(1-4.*mn_**2/s)**0.5/s/4.
    pre *= 2.*math.pi #from integration over phi angle
    tau = s/4./mn_**2
    #c1,c2,c3,c4 = initialize_tl(c1Re_,c1Im_,c2Re_,c2Im_,c3Re_,c3Im_,c4Re_,c4Im_)
    GM = F1N(s)+F2N(s)
    GE = F1N(s)+tau*F2N(s)
    form =8./3.*abs(GM)**2+4./3./tau*abs(GE)**2 #factors of 8/3, 4/3 coming from integration over theta
    return pre*form*par.gev2nb


# Decay rate of mediator-> pp + nn 
def GammaDM(mMed):
    Gammatot = 0
    for i in range(0,2):
        Gammatot+=GammaDM_mode(mMed,i)
    return Gammatot




