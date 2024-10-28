# Library to load
import math,cmath,os
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import numpy as np
from src.functions import alpha,Resonance
import src.pars as par


#
#####################################################################################
## Parameter set for hadronic current, parametrization taken from arXiv:1911.11147 ##
#####################################################################################
#
# masses and width from PDG
mKp  = 0.493677
mK0  = 0.497648
mpi0 = 0.1349766
mpip = 0.13957018
mKS  = 0.8956
gKS  = 0.047

# masses and amplitudes, I=0, phi resonances
isoScalarMasses  = [1019.461*par.MeV,1633.4*par.MeV,1957*par.MeV]
isoScalarWidths  = [   4.249*par.MeV, 218*par.MeV,  267*par.MeV]
isoScalarAmp   = [0.  ,0.233  ,0.0405  ]
isoScalarPhase = [0,1.1E-07,5.19]
# masses and amplitudes, I=1
isoVectorMasses = [775.26*par.MeV,1470*par.MeV,1720*par.MeV]#,1900*Resonance.MeV]
isoVectorWidths = [149.1 *par.MeV, 400*par.MeV, 250*par.MeV]#,100*Resonance.MeV]
isoVectorAmp   = [-2.34  ,0.594  ,-0.0179  ]
isoVectorPhase = [0,0.317,2.57]
# K* K pi coupling
#g2=math.sqrt(6.*math.pi*mKS**2/(0.5*mKS*Resonance.beta(mKS**2,mKp,mpip))**3*gKS)
g2=5.37392360229
# masses for the integrals
M=0.
m1=0.
m2=0.
m3=0.
#####################################################################################

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
hadronic_interpolator_0 = None
hadronic_interpolator_1 = None
hadronic_interpolator_2 = None
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
    readHadronic_Current()
    
#
#############################
## form factor calculation ##
#############################
#
# add up all isospin 0/1 components
def isoSpinAmplitudes(shat) :
    # I=0
    A0=0.
    for ix in range(0,len(isoScalarMasses)) :
        A0 +=cS_*isoScalarAmp[ix]*cmath.exp(complex(0.,isoScalarPhase[ix]))*\
              Resonance.BreitWignerFW(shat,isoScalarMasses[ix],isoScalarWidths[ix])
    # I=1
    A1=0.
    for ix in range(0,len(isoVectorMasses)) :
        A1 +=cI1_*isoVectorAmp[ix]*cmath.exp(complex(0.,isoVectorPhase[ix]))*\
              Resonance.BreitWignerFW(shat,isoVectorMasses[ix],isoVectorWidths[ix])
    return (A0,A1)


def readHadronic_Current():
    for imode in range(0,3):
        [energies, integral_values] = np.load(os.path.dirname(os.path.abspath(__file__))+"/KKpi/KKpi_coefficients_%d.npy" %imode,allow_pickle=True)
        integrals = {}
        for xen in range(0,len(energies)):
            integrals[energies[xen]] = integral_values[xen]
        x = []
        y = []
        for energy in energies:
            x.append(energy)
            s = energy**2
            I = integrals[energy]
            pre = 4.*g2**2
            # amplitudes
            A0,A1 = isoSpinAmplitudes(s)
            amp_12  = 0.
            amp_23  = 0.
            # Used A0, A1 relations like in 1010.4180, although irrelevant since I1_amp's take care of sign
            # K_L K_S pi0
            if(imode==0) :
                #amp_12  = 1./math.sqrt(6.)*(A0-A1)
                amp_12  = 1./math.sqrt(6.)*(A0+A1)
                amp_23  = amp_12
            # K+ K- pi0
            elif(imode==1) :
                #amp_12  = 1./math.sqrt(6.)*(A0+A1)
                amp_12  = 1./math.sqrt(6.)*(A0-A1)
                amp_23  = amp_12
            # K+pi-K0
            elif(imode==2) :
                # as two charge modes
                pre *=2
                amp_12 = 1./math.sqrt(6.)*(A0+A1)
                amp_23 = 1./math.sqrt(6.)*(A0-A1)
            # put everything together
            Itotal = I[0]*abs(amp_12)**2+I[1]*abs(amp_23)**2+2.*(I[2]*amp_12*amp_23.conjugate()).real
            y.append(pre*Itotal)
        global hadronic_interpolator_0,hadronic_interpolator_1,hadronic_interpolator_2
        if imode==0: hadronic_interpolator_0 = interp1d(x, y, kind='cubic',fill_value="extrapolate")
        if imode==1: hadronic_interpolator_1 = interp1d(x, y, kind='cubic',fill_value="extrapolate")
        if imode==2: hadronic_interpolator_2 = interp1d(x, y, kind='cubic',fill_value="extrapolate")

#
###############
## processes ##
###############
#

# cross-section for e+e- -> KKpi for certain mode
def sigmaSM_mode(s,imode):
    if s<=(2*mK0+mpip)**2: return 0
    en = math.sqrt(s)
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    pre*= 1./math.sqrt(s)
    if imode==0: had = abs(hadronic_interpolator_0(en))
    if imode==1: had = abs(hadronic_interpolator_1(en))
    if imode==2: had = abs(hadronic_interpolator_2(en))
    return pre*had*par.gev2nb

# cross-section for e+e- -> KKpi total
def sigmaSM(s):
    sigmatot = 0.
    for imode in [0,1,2]:
        sigmatot += sigmaSM_mode(s,imode)
    return sigmatot
            
# Dark 

# Decay rate of mediator-> KKpi for certain mode
def GammaDM_mode(mMed,imode):
    if mMed**2<=(2*mK0+mpip)**2: return 0
    # vector spin average
    pre = 1/3.
    #phase space
    pre *= 1.
    if imode==0: had = abs(hadronic_interpolator_0(mMed))
    if imode==1: had = abs(hadronic_interpolator_1(mMed))
    if imode==2: had = abs(hadronic_interpolator_2(mMed))
    return pre*had

# Decay rate of mediator-> KKpi total
def GammaDM(mMed):
    Gammatot = 0
    for i in range(0,3):
        Gammatot+=GammaDM_mode(mMed,i)
    return Gammatot

# cross-section for e+e- -> KKpi for certain mode
def sigmaDM_mode(s,imode):
    if s<=(2*mK0+mpip)**2: return 0
    en = math.sqrt(s)
    # Dark prefactor
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre = DMmed2*s*(1+2*mDM_**2/s)/3.
    #phase space
    pre*= 1./math.sqrt(s)
    if imode==0: had = abs(hadronic_interpolator_0(en))
    if imode==1: had = abs(hadronic_interpolator_1(en))
    if imode==2: had = abs(hadronic_interpolator_2(en))
    return pre*had*par.gev2nb

# cross-section for e+e- -> KKpi total
def sigmaDM(s):
    if s<=(2*mK0+mpip)**2: return 0
    sigDM = 0.
    for i in range(0,3):
        sigDM = sigmaDM_mode(s,i)
    return sigDM
