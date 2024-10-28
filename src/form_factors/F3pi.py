# Libraries to load
import math,cmath,os
from scipy.interpolate import interp1d
from scipy import integrate
import numpy as np
from src.functions import alpha,Resonance
import src.pars as par

# define complex number
ii = complex(0.,1.)

#
#############################################
## parametrization based on hep-ph/0512180 ##
#############################################
#
# To speed up the calculation, the hadronic current has already been integrated over
# the phase space for the rho, omega, and phi contribution. The interference terms are included
# in that calculation. The results are stored in src/form_factors/3pi/ .
#############################################


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
cVector_ = [cI1_,cI0_,cS_]
coeffs = {}
hadronic_interpolator = None
###############################


###############################
# function to reset parameters#
###############################
def resetParameters(gDM,mDM,mMed,wMed,cMedu,cMedd,cMeds) :
    global gDM_,mDM_,mMed_,wMed_, cI1_, cI0_, cS_
    global cVector_
    cI1_ = cMedu-cMedd
    cI0_ = 3*(cMedu+cMedd)
    cS_ = -3*cMeds
    cVector_ = [cI1_,cI0_,cS_]
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    readHadronic_Current()

#
#############################
## form factor calculation ##
#############################
#

def hadronic_current(energy):
    s=energy**2
    pre= 1./(2*math.pi)**3/32./s**2
    total=0
    for ix in range(0,3):
        for jx in range(0,3):
            total+=pre*cVector_[ix]*cVector_[jx]*coeffs[energy][ix,jx]
    return total

def readHadronic_Current():
    [energies, coefficients] = np.load(os.path.dirname(os.path.abspath(__file__))+"/3pi/3pi_coefficients.npy",allow_pickle=True,encoding ="latin1")
    global coeffs
    for xen in range(0,len(energies)):
        coeffs[energies[xen]] = coefficients[xen]
    x = []
    y = []
    for energy in energies:
        x.append(energy)
        y.append(hadronic_current(energy))
    global hadronic_interpolator
    hadronic_interpolator = interp1d(x, y, kind='cubic',fill_value="extrapolate")


#
###############
## processes ##
###############
#

# Decay rate of mediator-> 3pions
def GammaDM(medMass):
    if medMass<3*par.mpi_: return 0
    pre = medMass/3.
    had = abs(hadronic_interpolator(medMass))
    return pre*had

# cross section for DM annihilations
def sigmaDM(s):
    if s<(3*par.mpi_)**2: return 0
    en= math.sqrt(s)
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre= 1/3.*DMmed2*s*(1+2*mDM_**2/s)
    had = abs(hadronic_interpolator(en))
    return pre*had*par.gev2nb

# cross-section for e+e- -> pi+pi-pi0
def sigmaSM(s):
    if s<(3*par.mpi_)**2: return 0
    en = math.sqrt(s)
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    had = abs(hadronic_interpolator(en))
    return pre*had*par.gev2nb
