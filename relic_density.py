# Libraries to load
import numpy as np
from numpy import genfromtxt
from tqdm.auto import  trange

import os,math,time
import src.pars as par
import scipy
from scipy.special import kn
from scipy.interpolate import interp1d
from scipy.integrate import quad,odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from src.vecdecays import Model
import src.utilities as usefunc
import src.width_inel as winel

from pathos.multiprocessing import ProcessingPool as Pool
import warnings

 
    
class Cosmology:

    #relativistic degrees of freedoms, we assume g*s=g*
    def geff_s(self,x):
        dof_values = genfromtxt(os.path.dirname(__file__)+'/src/data/smdof.dat', delimiter=',',skip_header=1).T
        temp = dof_values[0]
        geff = dof_values[1]**2
    
        geffs = interp1d(temp, geff, kind='cubic',fill_value="extrapolate")
        return abs(geffs(self.mDM2/x))
    
    # Lambda factor in Boltzman equation as in my notes reimitz_d.pdf (2.21), Planckmass absorbed in y_eq
    def lambda_factor(self,x):
        return 0.264246*self.mDM2*self.geff_s(x)**0.5*par.mPlanck        
    
    def Hubble(self,x):
        return (1.66*self.geff_s(x)**0.5*(self.mDM2/x)**2)/par.mPlanck   

    def nequil(self, x, meq, dof):
        xeq = x*meq/self.mDM2     
        if xeq>=1:
            # neq = dof*meq**3*kn(2,xeq)/(2*math.pi**2*xeq)
            neq = dof*(meq*(self.mDM2/x)/2/math.pi)**(3/2.)*np.exp(-xeq)
        else:
            neq = dof*3*(1.2)*(self.mDM2/x)**3/(4*math.pi**2)
        return neq
    
    def entropyS(self,T):
        return 2*math.pi**2*self.geff_s(self.mDM2/T)*T**3/45
    
    # equilibrium yield
    def eq_yield_new(self, x , meq = None, dof = None):
        mpart = meq if meq is not None else self.mDM2
        dofpart = dof if dof is not None else self.dof
        return self.nequil(x, mpart, dofpart)/self.entropyS(self.mDM2/x)

    def eq_yield(self,x):
        return 0.145*(self.dof/self.geff_s(x))*x**1.5*math.exp(-x)



class CrossSections(Model,Cosmology):

    def __init__(self, name, coup, DMtype, split=0., Rrat=3, mhad= None, dof = 2):
        super().__init__(name, coup, DMtype, split, Rrat, mhad)

        # define the degrees of freedom
        self.dof = dof
        
    def set_DM(self,mDM=1.,gQ=1.,gDM=1.):
        self.mDM1 = mDM
        self.mDM2 = mDM*(self.split+1)
        self.mMed = self.mDM1*self.Rrat
        self.gQ = gQ
        self.gDM = gDM
        self.alphaDM = self.gDM**2/4./math.pi

        #compute widths
        self.calcwid(self.gQ,self.gDM)
            
               
    def kallen(self, a,b,c):
        return a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*c*b
    
    ######################
    ### Cross Sections ###
    #####################
    
    # 12 -> ff
    def xsec(self, s, nferm = 0):
        sqrts = math.sqrt(s)
        
        GammaVec = self.wsm['total'](self.mMed)+self.wdm['DM'](self.mMed)   
        
        if sqrts< 10:
            GammaSM = self.wsm['total'](sqrts) 
        else:
            GammaSM = self.normwidth(self.gQstates, sqrts, gQ = self.gQ, gD = self.gDM)
        
        GammaDM = self.normwidth(self.gDstates, sqrts, gQ = self.gQ, gD = self.gDM, mDMval=self.mDM1)
        # kinematics
        fac =  12*math.pi*s**2/self.kallen(s,self.mDM1**2,self.mDM2**2)

        return fac*GammaDM*GammaSM/((s-self.mMed**2)**2+self.mMed**2*GammaVec**2)
    
    ### 22 -> 11 (t+u contributions)
    def xsec_22_11(self,s, nferm = 0):
        m1 = self.mDM1
        m2 = self.mDM2
        mZ = self.mMed        
        
        fac = np.sqrt(-4*m1**2 + s)/(32*math.pi*mZ**4*s*np.sqrt(-4*m2**2 + s))

        fac1num = 2*(-mZ**2 + (m1 - m2)**2)**2*(2*mZ**4 + (m1 + m2)**4)
        fac1den = m1**4 - 2*m1**2*(m2**2 + mZ**2) + mZ**2*s + (m2**2 - mZ**2)**2
        
        fac2numA = 12*m1**8 - 16*m1**7*m2 - 16*m1**6*m2**2 - 40*m1**6*mZ**2 + 16*m1**5*m2**3 + 32*m1**5*m2*mZ**2 + 8*m1**4*m2**4 - 88*m1**4*m2**2*mZ**2 + 76*m1**4*mZ**4 + 16*m1**3*m2**5 + 192*m1**3*m2**3*mZ**2 - 112*m1**3*m2*mZ**4 - 16*m1**2*m2**6 - 88*m1**2*m2**4*mZ**2 + 264*m1**2*m2**2*mZ**4 - 80*m1**2*mZ**6 - 16*m1*m2**7 + 32*m1*m2**5*mZ**2 - 112*m1*m2**3*mZ**4 + 96*m1*m2*mZ**6 + 12*m2**8 - 40*m2**6*mZ**2 + 76*m2**4*mZ**4 - 80*m2**2*mZ**6 + 32*mZ**8 + 8*mZ**2*s**2*(-m1 + m2 + 2*mZ)*(m1 - m2 + 2*mZ) + 4*s*(12*mZ**6 - 4*mZ**4*(7*m1**2 - 2*m1*m2 + 7*m2**2) + mZ**2*(m1 - m2)**2*(9*m1**2 + 2*m1*m2 + 9*m2**2) - 3*(m1 - m2)**4*(m1 + m2)**2)
        fac2numB = np.arctanh((np.sqrt(-4*m1**2 + s)*np.sqrt(-4*m2**2 + s))/(2*m1**2 + 2*m2**2 - 2*mZ**2 - s)) 
        fac2den = np.sqrt(-4*m1**2 + s)*np.sqrt(-4*m2**2 + s)*(2*m1**2 + 2*m2**2 - 2*mZ**2 - s)
        
        fac3num = 8*mZ**2*s + (-2*mZ**2 + (m1 - m2)**2)**2
        
        return  self.gDM**4*fac*((fac1num/fac1den)-(fac2numA*fac2numB/fac2den)+fac3num)
        

    ########################
    ###  Thermal  Decay  ###
    ########################
    
    # decay 2 -> 1 SM SM 
    def thermalDec(self,x):
       
        widthC = winel.decayChi2(self.cfunc, self.Rrat)
        dec = widthC.width(states = "total", m1=self.mDM1, Delta=self.split, gChi=self.gDM, gQ=self.gQ)
        return (kn(1,x)/kn(2,x))*dec

    ########################
    ###  Kinetic  Rate   ###
    ########################

    def tdsigdtKIN(self,t, s, mf=0):
        m1 = self.mDM1
        m2 = self.mDM2
        mZ = self.mMed
        return t*(2*s**2 - 2*s*(m1**2 + m2**2 + 2*mf**2) + t**2 - t*(-2*s + (m1 - m2)**2) + 2*(m1*m2 + mf**2)**2)/(8*math.pi*(mZ**2 - t)**2*(m2**4 - 2*m2**2*mf**2 - 2*m2**2*s + mf**4 - 2*mf**2*s + s**2))
    
    def rateKINint1(self,s, mf=0):
        m2 = self.mDM2
        pf = (1/2.)*np.sqrt(m2**4 + (mf**2-s)**2 - 2*m2**2 * (mf**2+s) )/np.sqrt(s)
        return quad(self.tdsigdtKIN, -4*pf**2, 0 , args=(s, mf))[0]

    
    def rateKINint2(self,s,x, mf=0, mfac=1e+50):
        m2 = self.mDM2
        m1 = self.mDM1

        fac = -x/3./m1/m2
        pf = (1/2.)*np.sqrt(m2**4 + (mf**2-s)**2 - 2*m2**2 * (mf**2+s) )/np.sqrt(s)
        dpfds = (s**2 - (m2**2 - mf**2)**2)/4/s**(3./2)/np.sqrt(m2**4 + (mf**2-s)**2 - 2*m2**2 * (mf**2+s))
        
        d3pf = (4*math.pi)*(pf**2)*dpfds
        
        ef = np.sqrt(pf**2 + mf**2)
        dist = (np.exp(ef*x/self.mDM2)+1)**(-1)
        facv = pf/ef
        integrand = fac*d3pf*dist*(1-dist)*facv*self.rateKINint1(s)/(2*math.pi)**3
    
        return integrand*mfac
    
    def rateKIN(self, x, ferm, mfac=1e+50):
        cf = self.cferm[ferm](self.mMed)
        mf = par.mferm[ferm]
        fac = self.gDM**2*(self.gQ*cf)**2
        integral = quad(self.rateKINint2,(self.mDM2+mf)**2 , np.inf , args=(x,mf,mfac))[0]#,epsabs=1e-4,epsrel=1e-4)[0]
        return fac*integral/mfac


    ########################
    ### Thermal  Average ###
    ########################
        
    
    # 2f -> 1f :xsecv times Yf - Kin rate method
    def xsecvDMSM(self, x):
        
        scatdict = {key: self.cferm[key] for key in ["nue","numu","nutau","e"] if key in self.cferm}
        
        xsecvY = 0
        for key, val in scatdict.items():
            if abs(val(self.mMed)) > 1e-2:
                rate = self.rateKIN(x,ferm=key)
                xsecvY +=  rate/self.entropyS(self.mDM2/x)
                
        # no couplings w/neutrinos or electron (B mdoel for example)
        # factor 3 = colors -> one facor for xsec and another for nf
        if  xsecvY == 0:
            rateu = self.rateKIN(x,ferm="u")
            xsecvY += 3*3*rateu/self.entropyS(self.mDM2/x)

            rated = self.rateKIN(x,ferm="d")
            xsecvY += 3*3*rated/self.entropyS(self.mDM2/x)

        return xsecvY    
    
    
    def int_xsecv22(self, s, x, sigma, mi, mj, ferm = "e"):

        sqrts = math.sqrt(s)
        besselK1 = kn(1,sqrts*x/self.mDM2)
        xsec = sigma(s,ferm)
        
        kinfac= self.kallen(s,mi**2,mj**2)
            
        return kinfac*xsec*besselK1/sqrts
    
    # idfac = 2 for identical particles in the initial state 
    def xsecv22(self, x, sigma, mi, mj, ferm = "e", idfac = 1):
        
        mDM2 = self.mDM2
        T = mDM2/x
        
        nieq = self.nequil(x,mi,1)
        njeq = self.nequil(x,mj,1)
        fac = T/(32*math.pi**4*nieq*njeq)
        
        ixsec = quad(self.int_xsecv22, (mi+mj)**2, max((50*mDM2/x)**2, (150*mDM2)**2) ,
                     args=(x, sigma, mi, mj, ferm), points=[4*(mi+mj)**2,self.mMed**2,(2*self.mMed)**2])[0]

        return (fac/idfac)*ixsec    
    
   



class Boltzmann(CrossSections):
    
    def __init__(self, name, coup, DMtype, split=0., Rrat=3, mhad= None, dof = 2, mDM=1., gQ=1., gDM=1):
        
        # inhert variables from CrossSections class
        super().__init__(name, coup, DMtype, split, Rrat, mhad, dof)
        
        self.set_DM(mDM, gQ=1., gDM=1)
        
        ## parameter that controls if the Boltzmann equation will be computed with 
        ## interpolation of the thermal average xsecs
        self.interpolate = True
        
        warnings.filterwarnings("ignore", category= RuntimeWarning)
        
    def defBEQchi(self, logx, sol):

        x = np.exp(logx)
        W = sol
        
        eq_yield_chi = self.eq_yield(x*self.mDM1/self.mDM2)+self.eq_yield(x)
        Weq =  np.log(eq_yield_chi)
        
        lamb =  self.lambda_factor(x)
        xsecv_12ff = self.xsecv22(x, self.xsec, self.mDM1, self.mDM2)
        fac = self.nequil(x, self.mDM1, 2)*self.nequil(x, self.mDM2, 2)/(self.nequil(x, self.mDM1, 2)+self.nequil(x, self.mDM2, 2))**2
        
        try:
            p1 = -2*(lamb*xsecv_12ff/x)*fac*(np.exp(2*W[0])-np.exp(2*Weq))
            dWdlogx = np.exp(-W[0])*(p1)
            return np.array([dWdlogx])
        except:
            pass
        
               
    def solveBEQchi(self,xi=10.,xf=50.,step=50):
        #initial conditions
        W0 =  math.log(self.eq_yield(xi*self.mDM1/self.mDM2)+self.eq_yield(xi))
        Wsol = solve_ivp(self.defBEQchi,[np.log(xi),np.log(xf)], [W0],
                          method='Radau', vectorized=True,rtol=1e-3,atol=1e-5)
        return Wsol
    
    
    def relic_density_chi(self,xi=10.,xf=50.,step=50):
        self.boltzchi = self.solveBEQchi(xi,xf,step)
        self.relicchi = self.mDM1*np.exp(self.boltzchi.y[0,-1])*par.entropy0/par.rho_c
        print ('relic = ', self.relicchi ,'for mV=', self.mMed, ' and gQ=', self.gQ)
        return self.relicchi     

    # function to compute interpolated sigmav (to faster computation)
    def compINT(self):
        xarr = np.geomspace(1,200,600)
        
        xsecv12ff = []
        xsecv2211 = []
        xsecvDec = []
        xsecv2f1fY = []
        
        for x in xarr:
            # print ("x= ", x)

            ## 12 -> ff
            xv12ff = self.xsecv22(x, self.xsec, self.mDM1, self.mDM2)
            xsecv12ff.append(xv12ff)
            # print (xv12ff)

            ## 22 -> 11
            xv3 = self.xsecv22(x, self.xsec_22_11, self.mDM2,  self.mDM2, idfac= 2)
            xsecv2211.append(xv3)
            # print (xv3)
    
            ## decay
            dec4 = self.thermalDec(x)
            xsecvDec.append(dec4)
            # print (dec4)         
            
            # ( 2f -> 1f ) * Yf
            xscat = self.xsecvDMSM(x)
            xsecv2f1fY.append(xscat)
            
            # print (xscat)
            
        self.xv12ff_int = interp1d(xarr, xsecv12ff, fill_value="extrapolate")
        self.xv2211_int = interp1d(xarr, xsecv2211, fill_value="extrapolate")
        self.xvDec_int = interp1d(xarr, xsecvDec, fill_value="extrapolate")
        self.xv2f1fY_int = interp1d(xarr, xsecv2f1fY, fill_value="extrapolate")


    def defBEQ(self, logx, sol):
                     
        x = np.exp(logx)
        W1,W2 = sol
        
        W1eq =  np.log(self.eq_yield(x*self.mDM1/self.mDM2))
        W2eq = np.log(self.eq_yield(x))
        
        lamb =  self.lambda_factor(x)
        
        if self.interpolate:
            # interpolated xsecv
            xsecv_12ff = self.xv12ff_int(x)
            xsecv_2f1f_Y = self.xv2f1fY_int(x) 
            xsecv_2211 = self.xv2211_int(x)
            decrat = self.xvDec_int(x)

        else:
            xsecv_12ff = self.xsecv22(x, self.xsec, self.mDM1, self.mDM2)
            xsecv_2f1f_Y = self.xsecvDMSM(x)
            xsecv_2211 = self.xsecv22(x, self.xsec_22_11, self.mDM2,  self.mDM2, idfac= 2)
            decrat = self.thermalDec(x)
            
        try:
            p1 = -(lamb*xsecv_12ff/x)*(np.exp(W1+W2)-np.exp(W1eq+W2eq))
            
            pf = -(lamb*xsecv_2f1f_Y/x)*(np.exp(W1+W2eq-W1eq)-np.exp(W2))
            p2 = 2*pf # factor of 2 to include antifermions
        
            p3 = -2*(lamb*xsecv_2211/x)*(np.exp(2*W1+2*W2eq-2*W1eq)-np.exp(2*W2))
            
            p4 = -(decrat/self.Hubble(x))*(np.exp(W1+W2eq-W1eq) - np.exp(W2))
        
            dW1dlogx = np.exp(-W1)*(p1+p2+p3+p4)
            dW2dlogx = np.exp(-W2)*(p1-p2-p3-p4)
            
            return [dW1dlogx,dW2dlogx]
        except: pass
        
            
    def solveBEQ(self,xi=10.,xf=50.,step=50):
        # interpolated xsecv
        if self.interpolate: self.compINT()
        #initial conditions
        W0 = [math.log(self.eq_yield(xi*self.mDM1/self.mDM2)),math.log(self.eq_yield(xi))]
        Wsol = solve_ivp(self.defBEQ,[np.log(xi),np.log(xf)],W0,
                          method='Radau', vectorized=True,rtol=1e-3,atol=1e-5)
                         
        return Wsol
    
    def relic_density(self,xi=10.,xf=50.,step=50):            
        self.boltz = self.solveBEQ(xi,xf,step)
        self.relic = self.mDM1*np.exp(self.boltz.y[0][-1])*par.entropy0/par.rho_c
        print ('relic = ', self.relic ,'for mV=', self.mMed, ' and gQ=', self.gQ)
        return self.relic   

    
    def calc_rates(self,fermion= ["e", "nue", "numu", "nutau"], xlim= (1,200,200)):

            sol = self.boltz
            xval = np.exp(sol.t)  
            W1sol,W2sol = sol.y[0], sol.y[1]
            Y1sol,Y2sol = np.exp(np.asarray(W1sol)), np.exp(np.asarray(W2sol))
    
            self.Y1sol = interp1d(xval,Y1sol, fill_value="extrapolate")
            self.Y2sol = interp1d(xval,Y2sol, fill_value="extrapolate")       
            
            ch_kin = ['kin_' + ferm for ferm in fermion]
            channels = ['12ff','2211','dec'] + ch_kin
            
            xsecvval = {chan: [] for chan in channels}
            rateval =  {chan: [] for chan in channels+['12ff_eff']}
            
            xarr = np.geomspace(xlim[0],xlim[1],xlim[2])
            
            for x in xarr:
                neq2 = self.nequil(x, self.mDM2, 2)
                neq1 =  self.nequil(x, self.mDM1, 2)
                Hub = self.Hubble(x)
      
                n2val =  self.Y2sol(x)*self.entropyS(self.mDM2/x)
                n1val =  self.Y1sol(x)*self.entropyS(self.mDM2/x)
                ratn = n2val/n1val
                
                xv12ff = self.xsecv22(x, self.xsec, self.mDM1, self.mDM2)
                xsecvval['12ff'].append(xv12ff)
                rateval['12ff'].append(xv12ff*neq2/Hub)
                rateval['12ff_eff'].append((2*xv12ff*neq1*neq2/(neq1+neq2)**2)*(neq2+neq1)/Hub)                   
                 
                ## 22 -> 11
                # i wont put the identical factor since it cancels out with the 2 factor in the BEQ
                xv2211 = self.xsecv22(x, self.xsec_22_11, self.mDM2,  self.mDM2) #, idfac= 2)
                xsecvval['2211'].append(xv2211)
                rateval['2211'].append(xv2211*n2val*ratn/Hub)            
                                
                ## 2f -> 1f
                for chan in ch_kin:
                    fname = chan[4:]
                    ratekin = self.rateKIN(x,fname)
                    neqferm = self.nequil(x, par.mferm[fname], 2)
                    xsecvval[chan].append(ratekin/neqferm)
                    rateval[chan].append(ratekin*ratn/Hub)
        
                ## decay
                rdec = self.thermalDec(x)
                xsecvval['dec'].append(rdec)
                rateval['dec'].append(rdec*ratn/Hub)        
            
            self.xsecvval = usefunc.interp_dict(xsecvval, xarr)
            self.rateHval = usefunc.interp_dict(rateval, xarr)
            
            return self.xsecvval,self.rateHval

    
        
    def plot_FO(self, xlim = (1,200), ylim = (1e-16,1e-1), ptype = "Y1Y2", name=None):
        
        if ptype not in ['Y1Y2', 'Ychi', 'both']:
            raise ValueError("choice must be 'Y1Y2', 'Ychi', or 'both'")
        
        fig, ax = plt.subplots(figsize=(8,5.))
        
        colors = ['#184e77', '#184e77','#5AB5D6','#5AB5D6']
        
        # compute equilibrium yields
        xarr = np.geomspace(xlim[0],xlim[1],100) 
        Yeq1,Yeq2 = [],[]
        for x in xarr:
            Yeq1.append(self.eq_yield(x*self.mDM1/self.mDM2))
            Yeq2.append(self.eq_yield(x))
        
    
        if ptype in  ['Ychi', 'both']:
            sol = self.boltzchi
            xval = np.exp(sol.t)   
            Wsol = sol.y[0]
            Ysol = np.exp(np.asarray(Wsol))
            ax.plot(xval,Ysol, color = '#0DC6B3', ls= '-.',alpha=0.8,lw=1.5,
                    zorder =0)
            self.Ychi = interp1d(xval, Ysol, fill_value="extrapolate")
                               
            if ptype == 'Ychi':
                Yeq =  [a + b for a, b in zip(Yeq1, Yeq2)]
                ax.plot(xarr,Yeq)

            relic = self.relicchi
    
        if ptype in ['Y1Y2', 'both']:
            sol = self.boltz
            xval = np.exp(sol.t)  
            W1sol,W2sol = sol.y[0], sol.y[1]
            Y1sol,Y2sol = np.exp(np.asarray(W1sol)), np.exp(np.asarray(W2sol))
            
            ax.plot(xval,Y1sol, color = colors[1],lw=1.8, label= "$Y_1$", zorder= 10)
            ax.plot(xarr,Yeq1,ls='--', color = colors[0],lw=1.6, label=  "$Y_1^{\, \\rm eq}$",zorder= 9)
            ax.plot(xval,Y2sol, color = colors[3],lw=1.8, label= "$Y_2$",zorder= 8)
            ax.plot(xarr,Yeq2,ls='--', color = colors[2],lw=1.6, label=  "$Y_2^{\, \\rm eq}$",zorder= 7)           

            relic = self.relic
             
        textstr = '\n'.join(( r'$m_1  = %.2f \, {\rm GeV}$' % (self.mDM1, ),
                             r'$ g_Q = $ %s' % (usefunc.sci_not(self.gQ), ),
                             r'$ \alpha_D  = %.2f$' % (self.alphaDM, ),
                             r'$R  = %s$' % (self.Rrat, ),
                             r'$\Omega h^2 =  %.2f$' % (relic, )))
        
        plt.annotate(textstr, xy=(0.03, 0.05), xycoords='axes fraction',fontsize=14,
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5, edgecolor='none'))

        if ptype in ['Ychi', 'both']: ax.plot(0,0, color = '#0DC6B3', ls= '-.',alpha=0.8,lw=1.5,label=  "$Y^{\, \\rm eff}$", zorder =-10)
        plt.legend(ncol=3,loc= "best",fontsize=12)
        ax.set_ylim(ylim) 
        ax.set_xlim(xlim) 
        plt.yscale("log")
        plt.xscale("log")
        plt.title(self.modelname.replace("_", " ") + " + " +self.DMlab.replace("_", " ").replace("delta", "$\\Delta =$"), fontsize=16)
        plt.xlabel("$x= m_2/T$",fontsize = 18)
        plt.ylabel("$Y=\\dfrac{n}{s}$",fontsize = 18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.minorticks_on()
                

        if name:
            relicfold = usefunc.create_folder(self.modfolder+'/relic')
            self.plotlab = self.modelname.replace(" ","_")+"_"+self.DMlab+"_R_%s_gQ_%s_aDM_%.1f" % (self.Rrat,self.gQ, self.alphaDM)

            plt.savefig(relicfold+'/fo_curve_'+self.plotlab+'_'+name+".pdf", bbox_inches = "tight")
    
            filename = relicfold + '/fo_data_'+self.plotlab+'_'+name+'.txt'
            with open(filename, 'w') as txtfile:
                txtfile.write("# x \t Y1 \t Y2 \t Ychi \t relic=%.5f \n" % (relic))
                for i in range(0,len(xval)):
                    if ptype == 'both':
                        txtfile.write("%s \t %s \t %s \t %s \n" %(xval[i], Y1sol[i], Y2sol[i], self.Ychi(np.asarray(xval))[i] ))
                    elif ptype == 'Ychi': 
                        txtfile.write("%s \t - \t - \t %s \n" %(xval[i], self.Ychi(np.asarray(xval))[i] ))
                    elif ptype == 'Y1Y2': 
                        txtfile.write("%s \t %s \t %s \t - \n" %(xval[i], Y1sol[i], Y2sol[i]))
                txtfile.close()


    ####################################
    ###  thermal target computation  ###
    ####################################
    
        
    def compLinear(self, rdsarr, epsarr):
        rdslog = np.log(rdsarr)
        epslog = np.log(epsarr)
        coeff = np.polyfit(rdslog, epslog, 1)
        self.coeff = coeff
    
    
    def relicF(self, x):
        return np.exp(self.coeff[1]) * x ** self.coeff[0]
    
    def compRelic(self, mdm, eps, step =100):
        # global boltzZQ
        self.set_DM(mDM=mdm,gQ=eps,gDM=self.gDM)
        rds = self.relic_density_chi(xi=10.,xf=100.,step=step)
        return rds
    
    
    def run_segment(self,segment):
        
        self.omega = 0.12
        self.omega_dn = 0.12 - (2*0.0012)
        self.omega_up = 0.12 + (2*0.0012)
        
        self.mVarrTT = []
        self.epsarrTT = []
        self.rdsarrTT = []       
        
        self.start_gq = 1e-3
        initEps = segment[0]*self.start_gq
    
        path = "txt_targets/countor_%s_%s_del_%s_aD_%s_R_%s_file_%s_seg_%.3f.txt" %(self.modelname,
                                                                            self.DM,
                                                                            self.split,
                                                                            self.alphaDM,
                                                                            self.Rrat,
                                                                            self.ttype,
                                                                            segment[-1])
        
        mVs = segment
        mDMs = mVs/self.Rrat
        start_time = time.time()
        for i in trange(len(mVs)):
            print("mV = %f --- %s seconds ---" % (mVs[i],(time.time() - start_time)))
            
            eps = initEps
            rds = self.compRelic(mDMs[i], eps)
            
            eps2 = initEps*1.1
            rds2 = self.compRelic(mDMs[i], eps2)
                
            self.compLinear([rds,rds2], [eps,eps2] )   
            epsNew = self.relicF(self.omega)
            rdsNew = self.compRelic(mDMs[i], epsNew)
            
            if rdsNew < self.omega_dn:
                epsNew = self.relicF(self.omega_up)
                rdsNew = self.compRelic(mDMs[i], epsNew)
        
            if rdsNew > self.omega_up:
                epsNew = self.relicF(self.omega_dn)
                rdsNew = self.compRelic(mDMs[i], epsNew)
            
            fac = 1
            count = 1
            parr =[]
            while rdsNew> self.omega_up or rdsNew<self.omega_dn:
                prop = rdsNew-self.omega
                epsF =  abs(self.relicF(self.omega_up)-self.relicF(self.omega_dn))/self.relicF(self.omega_up)
        
                epsNew = epsNew*(1 +np.sign(prop)*fac*epsF) 
                rdsNew = self.compRelic( mDMs[i], epsNew)
                
                parr.append(np.sign(prop))
                count +=1
                if count>3:
                    
                    if parr[-1]*parr[-2] == -1:
                        fac = 0.5
                        
                    if parr[-1]*parr[-2] == 1 and abs(prop)>0.02:
                        fac = 2*count/4
    
                    continue
                   
            if self.omega_dn<rdsNew<self.omega_up:
                self.mVarrTT .append(mVs[i])
                self.epsarrTT.append(epsNew)
                self.rdsarrTT.append(rdsNew)
                
                # if you want to save the FO plots
                #boltzZQ.plot_FO(xlim = (1,100),ptype = "Ychi", name = "thermal_target")
                with open(path, 'a') as file:
                    file.write("%s \t %s \t %s \n" %(mVs[i], epsNew, rdsNew))
                
                initEps = epsNew
    
    def compute_target(self, mVi = 1e-2, mVf= 10, nmV= 30):# , start_gq = 1e-3, fileN= "scan"):
        mVs = np.geomspace(mVi, mVf, nmV)    
        self.ttype = "total"
        self.run_segment(mVs)
     
    
    
    ########################
    ###  parallelization ###
    ########################
    
    def merge_runs(self, fileN):
        
        folder = "txt_targets/"
        output = folder+ "countor_%s_%s_del_%s_aD_%s_R_%s_%s.txt" %(self.modelname,self.DM,self.split,self.alphaDM,self.Rrat,fileN)    
        all_rows = []
    
        for filename in os.listdir(folder):
            if filename.split("_")[-3] == "par" and filename.split("_")[-2] == "seg":
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as infile:
                        for line in infile:
                            row = list(map(float, line.strip().split()))
                            all_rows.append(row)
                    
                    os.remove(file_path)
    
        # Sort all rows by the first column
        all_rows.sort(key=lambda x: x[0])
    
        # Write sorted rows to output file
        with open(output, 'w') as outfile:
            outfile.write(" mZQ [GeV]  \t  gQ   \t   Omega h^2  \n")
            for row in all_rows:
                outfile.write(" ".join(map(str, row)) + "\n")
            
            
    def parallel_run(self, mass_values, num_segments):
        segments = np.array_split(mass_values, num_segments)
        with Pool() as pool:
            pool.map(self.run_segment, segments)
        
        
    
    def parallel_tharget(self, mVi = 1e-2, mVf= 10, nmV= 30 , nump = 10, fileN= "target"):
        mVs = np.geomspace(mVi, mVf, nmV) 
        massZQ = mVs
        num_processes = nump 
        self.ttype = "par"
        self.parallel_run(massZQ, num_processes)
        self.merge_runs(fileN)
       



