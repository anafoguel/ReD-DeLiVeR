# Libraries to load
import math,random,glob,os
import numpy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from src.functions import Resonance,alpha
import src.pars as par

#
###################################################################
## parametrization taken from 0804.0359 with fit from 1911.11147 ##
###################################################################
#
# To speed up the calculation, the phase space calculation has been
# performed already. The results are stored in src/form_factors/4pi/. 
#############################################

# # masses and widths from PDG (fixed for the fit)
mpip = .13957061
mpi0 = .1349770
mRho  = .7755
gRho  = .1494
mRho1 =1.459
gRho1 =0.4
mRho2 =1.72
gRho2 =0.25
ma1=1.23
ga1=.2
mf0=1.35
gf0=0.2
mOmega=.78265
gOmega=.00849

# as given in 0804.0359
g_omega_pi_rho=42.3
g_rho_pi_pi=5.997
g_rho_gamma=.1212

# # fit parameters of own fit
c_f0     = 124.10534971287902
beta1_f0 = 73860.28659732222
beta2_f0 = -26182.725634782986
beta3_f0 = 333.6314358023821

c_omega     = -1.5791482789120541
beta1_omega = -0.36687866443745953
beta2_omega =  0.036253295280213906
beta3_omega = -0.004717302695776386

c_a1     = -201.79098091602876
beta1_a1 = -0.051871563361440096
beta2_a1 = -0.041610293030827125
beta3_a1 = -0.0018934309483457441

c_rho = -2.3089567893904537

mBar1 = 1.437
mBar2 = 1.738
mBar3 = 2.12

gBar1 = 0.6784824438511003
gBar2 = 0.8049287553822373
gBar3 = 0.20919646790795576

br_omega_pi_gamma = 0.084
###################################################################

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
coeffs_neutral={}
coeffs_charged={}
omega_interpolator = None
hadronic_interpolator_n = None
hadronic_interpolator_c = None
###############################

###############################
# function to reset parameters#
###############################
def resetParameters(gDM,mDM,mMed,wMed,cMedu,cMedd,cMeds) :
    global cI1_,cI0_,cS_
    global cRhoOmPhi_
    global gDM_,mDM_,mMed_,wMed_
    gDM_ = gDM
    mDM_ = mDM
    mMed_ = mMed
    wMed_ = wMed
    cI1_ = cMedu-cMedd
    cI0_ = 3*(cMedu+cMedd)
    cS_ = -3*cMeds
    readHadronic_Current()
    
###############################

#
#############################
## form factor calculation ##
#############################
#

def Frho(Q2,beta1,beta2,beta3) :
    return 1./(1+beta1+beta2+beta3)*(       Resonance.BW3(Q2,mRho ,gRho )
                                     +beta1*Resonance.BW3(Q2,mBar1,gBar1)
                                     +beta2*Resonance.BW3(Q2,mBar2,gBar2)
                                     +beta3*Resonance.BW3(Q2,mBar3,gBar3))
def contributions(Q2):
    # coefficients for the cross section
    coeffs=numpy.zeros((4,4),dtype=complex)
    for i in range(0,4) :
        f1 = 0.
        if i==0 :
            f1 = c_a1*Frho(Q2,beta1_a1,beta2_a1,beta3_a1)
        elif i==1 :
            f1 = c_omega*Frho(Q2,beta1_omega,beta2_omega,beta3_omega)
        elif i==2 :
            f1 = c_f0   *Frho(Q2,beta1_f0   ,beta2_f0   ,beta3_f0   )
        elif i==3 :
            f1 = c_rho
        for j in range (0,4) :
            f2 = 0.
            if j==0 :
                f2 = c_a1   *Frho(Q2,beta1_a1,beta2_a1,beta3_a1)
            elif j==1 :
                f2 = c_omega*Frho(Q2,beta1_omega,beta2_omega,beta3_omega)
            elif j==2 :
                f2 = c_f0   *Frho(Q2,beta1_f0   ,beta2_f0   ,beta3_f0   )
            elif j==3 :
                f2 = c_rho
            coeffs[i,j]=f1*numpy.conj(f2)
    return coeffs

# phase space + form factors
def hadronic_current(Q2,npoints,wgt,wgt2,omegaOnly=False) :
    # contributions from several subsequent processes
    coeffs = contributions(Q2)
    # compute the cross section and error
    total = complex(0.,0.)
    toterr= complex(0.,0.)
    if(not omegaOnly) :
        for i1 in range(0,4) :
            for i2 in range (0,4) :
                total += cI1_**2*wgt[i1,i2]*coeffs[i1,i2]
                for j1 in range(0,4) :
                    for j2 in range (0,4) :
                        toterr += cI1_**4*coeffs[i1,i2]*coeffs[j1,j2]*wgt2[i1,i2,j1,j2]
    else :
        total  += cI1_**2*wgt[1,1]*coeffs[1,1]
        toterr += cI1_**4*coeffs[1,1]*coeffs[1,1]*wgt2[1,1,1,1]
    toterr = math.sqrt((toterr.real-total.real**2)/npoints)
    return total, toterr

def readHadronic_Current():
    global hadronic_interpolator_n
    global hadronic_interpolator_c
    readCoefficients()
    # neutral: pi+pi-2pi0
    x = []
    y = []
    for (key,val) in sorted(coeffs_neutral.items()) :
        en = key
        s=en**2
        x.append(en)
        (npoints,wgt,wgt2) = val
        hadcurr,hadcurr_err = hadronic_current(s,npoints,wgt,wgt2,omegaOnly=False)
        y.append(abs(hadcurr))
    hadronic_interpolator_n = interp1d(x, y, kind='cubic',fill_value=(0.,0.))
    # charged: 2pi+2pi-
    x = []
    y = []
    for (key,val) in sorted(coeffs_charged.items()) :
        en = key
        s=en**2
        x.append(en)
        (npoints,wgt,wgt2) = val
        hadcurr,hadcurr_err = hadronic_current(s,npoints,wgt,wgt2,omegaOnly=False)
        y.append(abs(hadcurr))
    hadronic_interpolator_c = interp1d(x, y, kind='cubic',fill_value=(0.,0.))
    
#
###############
## processes ##
###############
#

def GammaDM_mode(mMed,mode) :
    if cI1_==0: return 0
    if mode==0: 
        if mMed<0.85: return 0
    if mode==1: 
        if mMed<0.6125: return 0
    # m4Pi_ = 4*mpip
    pre = 1./3
    pre *=(2.*math.pi)**4/2./mMed
    hadcurr = 0
    if mode==0: hadcurr = hadronic_interpolator_n(mMed)
    if mode==1: hadcurr = hadronic_interpolator_c(mMed)   
    # print (pre, hadcurr)
    return pre*abs(hadcurr)

def GammaDM(mMed):
    Gammatot = 0
    for mode in [0,1]:
        Gammatot+=GammaDM_mode(mMed,mode)
    return Gammatot

def sigmaDM(s,mode) :
    if cI1_==0: return 0
    if mode==0: 
        if s<(0.85)**2: return 0
        m4Pi_ = 2*mpip + 2*mpi0
    if mode==1: 
        if s<(0.6125)**2: return 0
        m4Pi_ = 4*mpip
    if s<(m4Pi_)**2: return 0
    # leptonic part contracted
    sqrts = math.sqrt(s)
    cDM = gDM_
    DMmed = cDM/(s-mMed_**2+complex(0.,1.)*mMed_*wMed_)
    DMmed2 = abs(DMmed)**2
    pre=DMmed2*s*(1+2*mDM_**2/s)/3.
    #prefactor of phase space
    pre *=(2.*math.pi)**4/2./s*par.gev2nb
    if mode==0: hadcurr = hadronic_interpolator_n(sqrts)
    if mode==1: hadcurr = hadronic_interpolator_c(sqrts)    
    return pre*hadcurr

def sigmaSM_mode(s,mode) :
    if cI1_==0: return 0
    if mode==0: 
        if s<(0.85)**2: return 0
        m4Pi_ = 2*mpip + 2*mpi0
    if mode==1: 
        if s<(0.6125)**2: return 0
        m4Pi_ = 4*mpip
    if s<(m4Pi_)**2: return 0
    # leptonic part contracted
    sqrts = math.sqrt(s)
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    #prefactor of phase space
    pre *=(2.*math.pi)**4/2./s*par.gev2nb
    if mode==0: hadcurr = hadronic_interpolator_n(sqrts)
    if mode==1: hadcurr = hadronic_interpolator_c(sqrts)    
    return pre*abs(hadcurr)

def sigmaSM(s):
    sigmatot = 0.
    for mode in [0,1]:
        sigmatot += sigmaSM_mode(s,mode)
    return sigmatot
    
###############
## Utilities ##
###############

def readCoefficients() :
    global coeffs_neutral,coeffs_charged
    if(len(coeffs_neutral)!=0) : return
    omega={}
    for fname in glob.glob(os.path.dirname(os.path.abspath(__file__))+"/4pi/*neutral*.dat"):
        output = readPoint(fname)
        coeffs_neutral[output[0]] = output[1]
        omega[output[0]] = output[1][1][1,1]
    for fname in glob.glob(os.path.dirname(os.path.abspath(__file__))+"/4pi/*charged*.dat"):
        output = readPoint(fname)
        coeffs_charged[output[0]] = output[1]
    x=[]
    y=[]
    for val in sorted(omega.keys()) :
        x.append(val)
        y.append(omega[val].real)
    global omega_interpolator
    omega_interpolator =  interp1d(x, y, kind='cubic',fill_value="extrapolate")
        
def readPoint(fname) :
    file=open(fname)
    line=file.readline().strip().split()
    energy = float(line[0])
    npoints = int(line[1])
    line=file.readline().strip()
    ix=0
    iy=0
    wgtsum  = numpy.zeros((4,4)    ,dtype=complex)
    while len(line)!=0 :
        if(line[0]=="(") :
            index = line.find(")")+1
            wgtsum[ix][iy] = complex(line[0:index])
            iy+=1
            line = line[index:]
        elif(line[0:2]=="0j") :
            wgtsum[ix][iy] = 0.
            iy+=1
            line = line[2:]
        else :
            print ('fails',line)
            quit()
            
        if(iy==4) :
            iy=0
            ix+=1
    line=file.readline().strip()
    ix1=0
    iy1=0
    ix2=0
    iy2=0
    wgt2sum = numpy.zeros((4,4,4,4),dtype=complex)
    while len(line)!=0 :
        if(line[0]=="(") :
            index = line.find(")")+1
            wgt2sum[ix1][iy1][ix2][iy2] = complex(line[0:index])
            iy2+=1
            line = line[index:]
        elif(line[0:2]=="0j") :
            wgt2sum[ix1][iy1][ix2][iy2] = 0.
            iy2+=1
            line = line[2:]
        else :
            print ('fails',line)
            quit()
        if(iy2==4) :
            iy2=0
            ix2+=1
            if(ix2==4) :
                ix2=0
                iy1+=1
                if(iy1==4) :
                    iy1=0
                    ix1+=1
    file.close()
    return (energy,[npoints,wgtsum,wgt2sum])