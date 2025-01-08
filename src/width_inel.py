import os, sys, inspect, math, collections
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import scipy



sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(
                inspect.getfile(inspect.currentframe()))), "src/"))

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(
                inspect.getfile(inspect.currentframe()))), ""))

import pars as par
from form_factors import FPiGamma_new
from form_factors import FK
from form_factors import F2pi

import vecdecays as vd


### pars

mpi = par.mpi0_
mK0 = par.mK0
mKp = par.mKp


###############################################################################
class DecayError(Exception):
    """
    Error for the 'decayChi2' class.
    """
    pass

###############################################################################
class decayChi2:
    """
    Calculate the 3 body decay  Chi2 -> Chi1 + SM + SM
    """
    ###########################################################################
    def __init__(self, coup, Rrat):
        """
        intialization
        
        R: mZd/m1 mass ratio [GeV]
        coup: U(1)_Q couplings to SM fermions
        """
        self.DM =  "inelastic"
        self.Rrat = Rrat
        self.coup = coup
        
        # couplings that are functions of the mediator mass mV
        self.cfunc = [entry if callable(entry) else (lambda value: lambda _: value)(entry) for entry in coup]

        self.cferm = {
                    "d":     self.cfunc[0],
                    "u":     self.cfunc[1],
                    "s":     self.cfunc[2],
                    "c":     self.cfunc[3],
                    "b":     self.cfunc[4],
                    "t":     self.cfunc[5],
                    "e":     self.cfunc[6],
                    "mu":    self.cfunc[7],
                    "tau":   self.cfunc[8],
                    "nue":   self.cfunc[9],
                    "numu":  self.cfunc[10],
                    "nutau": self.cfunc[11]
                    }

        self.wc2states = ["e_e", "mu_mu", "tau_tau","nue_nue", "numu_numu", "nutau_nutau","piGamma", "KK"]
        if (self.cferm["u"](1)- self.cferm["d"](1)) != 0: self.wc2states += ["pipi"]

        self.__cache = {}        
    

    ###########################################################################
    def normwidth(self, states, m1, Delta):
        """
        Calculate the width[GeV] for a given set of states 
        considering the following model parameters:

        m1:    Chi1 mass [GeV]
        Delta: (m2-m1)/m1  [unitless]
        gChi:   Zd Chi2 Chi2 dark coupling [unitless]
        gQ:  U(1)_Q gauge coupling  [unitless]
        """
        
        m2 =m1*(Delta+1)
        
        cMedu = self.cferm["u"](self.Rrat*m1)
        cMedd = self.cferm["d"](self.Rrat*m1)
        cMeds = self.cferm["s"](self.Rrat*m1)
        
        # Loop over the states.
        wtot = 0

        for state in (states,) if isinstance(states, str) else states:

            cache = self.__cache.get(state)
            if cache and cache[0] == m1 and cache[1] == Delta:
                #print (cache)
                wtot += cache[-1]; continue

            pname = state.split("_")

            if state == "total":
                wpart = self.normwidth(self.wc2states, m1, Delta)
                
            elif state == "leptons":
                wpart = self.normwidth(["e_e", "mu_mu", "tau_tau","nue_nue", "numu_numu", "nutau_nutau"], m1, Delta)
                # print ("wpart:",wpart,"wtot", wtot)
                
            elif state == "neutrinos":
                wpart = self.normwidth(["nue_nue", "numu_numu", "nutau_nutau"], m1, Delta)

            elif state == "charged":
                wpart = self.normwidth(["e_e", "mu_mu", "tau_tau", "piGamma"], m1, Delta)
                
            elif state == "quarks":
                wpart = self.normwidth(["u_u", "d_d", "s_s", "c_c","b_b", "t_t"], m1, Delta)

            # Decay into fermions of same flavour
            elif len(pname) == 2 and pname[0] == pname[1] and pname[0] in par.mferm:
                fname = pname[0]
                mf =  par.mferm[fname]
                xf = self.cferm[fname](self.Rrat*m1)
                if m2 > 2.0*mf+m1:
                    upp = (m2 - mf)**2
                    low = (m1 + mf)**2
                    intF = quad(intF3body2,low,upp,args=(m1,m2,mf))[0]
                
                    wpart = (xf**2)*(1/64/m2**3)*(1/(2*np.pi)**3)*intF
                    
                    # neutrinos have an extra factor of 1/2 since Zq couples only to LH nu 
                    if fname[0:2] == "nu":
                        wpart = wpart/2.
                        
                else: wpart = 0
                
            
            elif state == "piGamma":
                FPiGamma_new.resetParameters(1.,0.,0.,0.,cMedu,cMedd,cMeds)
                
                if m2 > mpi+m1:
                    upp = (m2)**2
                    low = (m1 + mpi)**2
                    intF = quad(intPiG3body2,low,upp,args=(m1,m2))[0]
                    # print (upp,low,intF)
                    wpart = (1/32/m2**3)*(1/(2*np.pi)**3)*intF
                       
                else: wpart = 0
                
                
            elif state == "KK":
                FK.resetParameters(1,0.,0.,0.,cMedu,cMedd,cMeds)
                                
                if m2 > m1 + 2*mKp:
                    mKaon = mKp
                    intF = quad(intKK3body2,(m1 + mKaon)**2,(m2 - mKaon)**2,args=(m1,m2,mKaon),epsabs=1e-1,epsrel=1e-2)[0]
                    wKK1 = (1/32/m2**3)*(1/(2*np.pi)**3)*intF
                    wpart = wKK1
                
                if m2 > m1 + 2*mK0:
                    mKaon = mK0
                    intF = quad(intKK3body2,(m1 + mKaon)**2,(m2 - mKaon)**2,args=(m1,m2,mKaon),epsabs=1e-1,epsrel=1e-2)[0]
                    wKK0 = (1/32/m2**3)*(1/(2*np.pi)**3)*intF
                    wpart += wKK0
                    
                else: wpart = 0
             
            elif state == "pipi":
                F2pi.resetParameters(1,0.,0.,0.,cMedu,cMedd,cMeds)

                if m2 > m1 + 2*mpi:
                    intF = quad(intPP3body2,(m1 + mpi)**2,(m2 - mpi)**2,args=(m1,m2),epsabs=1e-1,epsrel=1e-2)[0]
                    wpipi = (1/32/m2**3)*(1/(2*np.pi)**3)*intF
                    wpart = wpipi

                else: wpart = 0               
                
            else: raise DecayError(
                "Unknown state '%s'." % state)

            wtot += wpart
            self.__cache[state] = (m1, Delta, wpart)

            
        return wtot

    def width(self, states, m1, Delta, gChi, gQ):
       
        mVec = self.Rrat*m1
        return (gQ*gChi/mVec**2)**2*self.normwidth(states, m1, Delta)
    
    
    ###########################################################################
    def tau(self, m1, Delta, gChi, gQ):
        """
        Calculate the lifetime [seconds] considering 
        the following model parameters:

        m1:    Chi1 mass [GeV]
        Delta: (m2-m1)/m1  [unitless]
        gChi:   Zd Chi2 Chi2 dark coupling [unitless]
        gQ:  U(1)_Q gauge coupling  [unitless]
        """
        return par.hbar/self.width("total", m1, Delta, gChi, gQ)
   
  
    ###########################################################################
    def bfrac(self, states, m1, Delta, gChi, gQ):
        """
        Calculate the branching fraction for a given set of states  
        considering the following model parameters:

        m1:    Chi1 mass [GeV]
        Delta: (m2-m1)/m1  [unitless]
        gChi:   Zd Chi2 Chi2 dark coupling [unitless]
        gQ:  U(1)_Q gauge coupling  [unitless]  
        """
        num = self.width(states, m1, Delta, gChi, gQ)
        if num == 0: return 0.0
        den = self.width("total", m1, Delta, gChi, gQ)
        if den == 0: return 0.0
        return num/den

    
    ###########################################################################    
    def comp_norm_widths(self, states, Delta, m1arr):
        
        wlepnorm = []
        whadnorm = []
        winvnorm = []
        
        for m1 in m1arr:
            wlepnorm.append(self.normwidth(states= ["e_e", "mu_mu", "tau_tau"], m1=m1, Delta=Delta))
            whadnorm.append(self.normwidth(states= "piGamma", m1=m1, Delta=Delta))
            winvnorm.append(self.normwidth(states= 'neutrinos', m1=m1, Delta=Delta))
            
        self.wlepnormI = interpolate.intp1d(m1arr, wlepnorm, fill_value='extrapolate')
        self.whadnormI = interpolate.intp1d(m1arr, whadnorm, fill_value='extrapolate')
        self.winvnormI = interpolate.intp1d(m1arr, winvnorm, fill_value='extrapolate')
        self.wtotnormI = interpolate.intp1d(m1arr, np.asarray(wlepnorm)+np.asarray(whadnorm)+np.asarray(winvnorm), fill_value='extrapolate')
          
   
    
   ###########################################################################    
    def intTwidth1(self,m232,m122,m1,m2,fname, gQval, gDMval):
        
        mf = par.mferm[fname] 
        m12 = m1**2
        m22 = m2**2
        mf2 = mf**2
        amp = 16*m12*m122-16*m12*m22+8*m12*m232-16*m1*m2*m232-32*m1*m2*mf2-16*(m122**2)+16*m122*m22-16*m122*m232+32*m122*mf2+8*m22*m232-8*(m232**2)-16*(mf2**2)

        #mZq = self.mVec
        den = 1
        #den = (m232 - mZq**2)**2 + mZq**2*self.zqwidth(mZq)**2
        
        mV = np.sqrt(m232) #invariant mass mee
        Br = GammaVff(par.fferm[fname],gQval,mV,self.cferm[fname](mV),mf)/self.zqwidth(mV)
        #print (Br)
        
        return amp/den/Br


    def intTwidth2(self,m122,m1,m2,fname,gQval, gDMval):
        
        mf =  par.mferm[fname] 
        
        E2Star = (m122 - m1**2 + mf**2)/(2*np.sqrt(m122))
        E3Star = (m2**2 - m122 - mf**2)/(2*np.sqrt(m122))
        low = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mf**2) + np.sqrt(E3Star**2 - mf**2))**2
        upp = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mf**2) - np.sqrt(E3Star**2 - mf**2))**2
        intm232 = quad(self.intTwidth1,low,upp,args=(m122,m1,m2,fname,gQval,gDMval))[0]
        return intm232



    ###########################################################################    
    def run_med(self):
        
        model = vd.Model("Zq")
        model.set_charges(self.coup)
        model.gQ = 1 #gQ = epsilon 
        model.set_DMtype(DM="No",Rchi=self.Rrat,gDM=1,)
        widths = vd.Widths(model)    
        widths.calc(mmin=1e-5,mmax=10.0)
        self.widthsMed = widths
        

    ###########################################################################    
    def totwidth_ee(self, m1, Delta, gChi, gQ):

        m2 =m1*(Delta+1)
        self.mVec = self.Rrat*m1
        
        fname = "e"
        mf =  par.mferm[fname]
        xf = self.cferm[fname](self.mVec)
        
        
        ### Zq part
        self.widthsMed.set_coup(gQ_new = gQ, gDM_new = gChi)
        self.widthsMed.calc_part()
        self.widthsMed.calc_total()
        self.zqwidth = interpolate.interp1d(self.widthsMed.masses,self.widthsMed.wtotal , fill_value="extrapolate")
        
        if m2 > 2.0*mf+m1:
            upp = (m2 - mf)**2
            low = (m1 + mf)**2

            intF = quad(self.intTwidth2,low,upp,args=(m1,m2,fname,gQ,gChi))[0]

            wpart = (xf**2)*(1/64/m2**3)*(1/(2*np.pi)**3)*intF
            # print (pname, wpart)
        else: wpart = 0

        Gtot = (gQ*gChi/self.mVec**2)**2*wpart
        return Gtot
                    
    ###########################################################################    
    def totwidth(self, name, m1, Delta, gChi, gQ):
        
        
        m2 =m1*(Delta+1)
        self.mVec = self.Rrat*m1
        
        model = vd.Model(name, self.coup, "No", Delta, self.Rrat, 1.737)
        model.calcwid(gQ, gChi, mmin=1e-3 ,mmax=30.0, step=10000)
        self.zqwidth = model.wtot 
        
        nudict = {"nue" : self.cferm["nue"](self.mVec),"numu" :self.cferm["numu"](self.mVec), "nutau" : self.cferm["nutau"](self.mVec) } 
        # if the vector mediator does not couple to neutrinos, use electron width
        if all(value == 0 for value in nudict.values()):            

            fname = "e"
            mf = par.mferm[fname]
            xf = self.cferm[fname](self.mVec)
                     
            if m2 > 2.0*mf+m1:
                upp = (m2 - mf)**2
                low = (m1 + mf)**2

                intF = quad(self.intTwidth2,low,upp,args=(m1,m2,fname,gQ,gChi),epsabs=1e-1,epsrel=1e-2)[0]

                wpart = (xf**2)*(1/64/m2**3)*(1/(2*np.pi)**3)*intF
                # print (pname, wpart)
            else: wpart = 0
                
        else:
            
            nucoups = [(key, value) for key, value in nudict.items() if value != 0]

            fname = nucoups[0][0]

            mf = par.mferm[fname] # mass=0 for neutrinos
            xf = self.cferm[fname](self.mVec)
                         
            if m2 > 2.0*mf+m1:
                upp = (m2 - mf)**2
                low = (m1 + mf)**2

                intF = quad(self.intTwidth2,low,upp,args=(m1,m2,fname,gQ,gChi),epsabs=1e-1,epsrel=1e-2)[0]

                # neutrinos have an extra factor of 1/2 since Zq couples only to LH nu 
                wpart = (1/2.)*(xf**2)*(1/64/m2**3)*(1/(2*np.pi)**3)*intF
            else: wpart = 0
                
                
        Gtot = (gQ*gChi/self.mVec**2)**2*wpart
        return Gtot       


############ Auxiliary functions ############

############################################

def kallen(self, a,b,c):
    return a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*c*b


def GammaVff(Cf, g, m, xF, mF):
    if m<2*mF:
        return 0
    else:  
        pre = Cf*(g* xF)**2/12/math.pi
        kin = m*(1 + 2*(mF**2/m**2))*np.sqrt(1- 4*(mF**2/m**2))
        return pre*kin
    

############################################


def feynpi0g(MAp,gQ,cZApi):
    Mpi0 = mpi
    ee =  3.02822e-1
    return ((MAp**2 - Mpi0**2)*((cZApi**2*ee**2*gQ**2*MAp**4)/2. - cZApi**2*ee**2*gQ**2*MAp**2*Mpi0**2 + (cZApi**2*ee**2*gQ**2*Mpi0**4)/2.))/(48.*math.pi*abs(MAp)**3)



def intF3body1(m232,m122,m1,m2,mf):
    m12 = m1**2
    m22 = m2**2
    mf2 = mf**2
    return 16*m12*m122-16*m12*m22+8*m12*m232-16*m1*m2*m232-32*m1*m2*mf2-16*(m122**2)+16*m122*m22-16*m122*m232+32*m122*mf2+8*m22*m232-8*(m232**2)-16*(mf2**2)


def intF3body2(m122,m1,m2,mf):
    E2Star = (m122 - m1**2 + mf**2)/(2*np.sqrt(m122))
    E3Star = (m2**2 - m122 - mf**2)/(2*np.sqrt(m122))
    low = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mf**2) + np.sqrt(E3Star**2 - mf**2))**2
    upp = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mf**2) - np.sqrt(E3Star**2 - mf**2))**2
    intm232 = quad(intF3body1,low,upp,args=(m122,m1,m2,mf))[0]
    return intm232


# pi Gamma
def intPiG3body1(m232,m122,m1,m2):
    
    m12 = m1**2
    m22 = m2**2
    mpi2 = mpi**2
    fac1 = m232*(2*(m12-m122)*(m122-m22)+m232*((m1-m2)**2-2*m122)-m232**2)
    fac2 = 2*mpi2*(-(m12*(m122-m22+m232))+2*m1*m2*m232+(m22+m232)*(m122-m22+m232))
    fac3 = - (mpi2**2)*(-m12+2*m1*m2+m22+m232)
    fac = fac1+fac2+fac3
    form = FPiGamma_new.FPiGamma(m232)
    # print ( "m1=",m1,fac,abs(form)**2)
    return 0.5*fac*abs(form)**2
        
  
def intPiG3body2(m122,m1,m2):
    E2Star = (m122 - m1**2 + mpi**2)/(2*np.sqrt(m122))
    E3Star = (m2**2 - m122)/(2*np.sqrt(m122))
    low = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mpi**2) + np.sqrt(E3Star**2))**2
    upp = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mpi**2) - np.sqrt(E3Star**2))**2
    intm232 = quad(intPiG3body1,low,upp,args=(m122,m1,m2))[0]
    # print (low, upp, m122, intm232)
    return intm232
              

# 2 kaons
def intKK3body1(m232,m122,m1,m2,mKaon):
    m12 = m1**2
    m22 = m2**2
    mK2 = mKaon**2
    fac = m12**2-m12*(4*m122-2*m22+m232)-2*m1*m2*(m232-4*mK2)+4*m122**2 -4*m122*(m22-m232+2*mK2)+m22**2-m22*m232+4*mK2**2
    
    if mKaon== mK0: imode=0
    if mKaon== mKp: imode=1 
    form = FK.Fkaon(m232, imode)
    
    # print ( "m1=",m1,fac,abs(form)**2)
    return 0.5*fac*abs(form)**2
        
  
def intKK3body2(m122,m1,m2,mKaon):
    E2Star = (m122 - m1**2 + mKaon**2)/(2*np.sqrt(m122))
    E3Star = (m2**2 - m122 - mKaon**2)/(2*np.sqrt(m122))
    low = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mKaon**2) + np.sqrt(E3Star**2 - mKaon**2))**2
    upp = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mKaon**2) - np.sqrt(E3Star**2 - mKaon**2))**2
    intm232 = quad(intKK3body1,low,upp,args=(m122,m1,m2,mKaon))[0]
    # print (low, upp, m122, intm232)
    return intm232

## 2 pions
def intPP3body1(m232,m122,m1,m2):
    m12 = m1**2
    m22 = m2**2
    mp2 = mpi**2
    fac = m12**2-m12*(4*m122-2*m22+m232)-2*m1*m2*(m232-4*mp2)+4*m122**2 -4*m122*(m22-m232+2*mp2)+m22**2-m22*m232+4*mp2**2
    
    form = F2pi.Fpi(m232, 1)
    
    # print ( "m1=",m1,fac,abs(form)**2)
    return 0.5*fac*abs(form)**2
        
  
def intPP3body2(m122,m1,m2):
    E2Star = (m122 - m1**2 + mpi**2)/(2*np.sqrt(m122))
    E3Star = (m2**2 - m122 - mpi**2)/(2*np.sqrt(m122))
    low = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mpi**2) + np.sqrt(E3Star**2 - mpi**2))**2
    upp = (E2Star + E3Star)**2 - (np.sqrt(E2Star**2 - mpi**2) - np.sqrt(E3Star**2 - mpi**2))**2
    intm232 = quad(intPP3body1,low,upp,args=(m122,m1,m2))[0]
    # print (low, upp, m122, intm232)
    return intm232

