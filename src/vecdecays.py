# Libraries to load
import numpy as np
import scipy,math
import os,time,sys
import matplotlib
import matplotlib.pyplot as plt
import csv
from src.functions import alpha, Resonance
import src.pars as par
import src.chan as ch
import src.data as data
import src.utilities as usefunc
from scipy.interpolate import interp1d

import src.width_inel as winel

import pandas as pd

class Processes():
    ######################
    ### cross sections ###
    ######################
    
    # e+e- -> mu+mu- #
    def xSection2e2mu(self,Q2):
        sigma2e2mu = (4 * math.pi * alpha.alphaEM(Q2)**2 * par.gev2nb)/(3*Q2)
        return sigma2e2mu
    
    # perturbative decay into quarks
    #---------------  eq. (9.7) of PDG - Quantum Chromodynamics  -------------------# 
    
    def RInclusive(self,Q):
        cMedu, cMedd, cMeds, cMedc, cMedb = self.cferm["u"](Q), self.cferm["d"](Q), self.cferm["s"](Q), self.cferm["c"](Q), self.cferm["b"](Q)
        sumQq =cMedu+cMedd+cMeds
        sumQq2 = cMedu**2+cMedd**2+cMeds**2
        nf=3
        if (2*par.mD0)<=Q:
            sumQq += cMedc
            sumQq2 += cMedc**2
            nf += 1
        if (par.mUps)<Q:
            sumQq += cMedb
            sumQq2 += cMedb**2
            nf += 1
        nc = 3
        Rew = (nc * sumQq2)
        eta = sumQq**2/(3*sumQq2)
        coeff = [1.,1.9857 - 0.1152*nf,-6.63694 - 1.20013*nf - 0.00518*nf**2 - 1.240*eta,
             -156.61 + 18.775*nf - 0.7974*nf**2 + 0.0215*nf**3-(17.828 - 0.575*nf)*eta]
        lambQCD =0
        if Q>1.:
            for n in range(0,4):
                lambQCD += coeff[n]*(alpha.alphaQCD(Q)/math.pi)**(n+1)
        return Rew*(1+lambQCD)
   
    
    ####################
    ### decay widths ###
    ####################
    
    ### widths with SM final states ###
    # vector -> f fbar
    def GammaVff(self, Cf, g, m, xF, mF):
        if m<2*mF:
            return 0
        else:  
            pre = Cf*(g* xF)**2/12/math.pi
            kin = m*(1 + 2*(mF**2/m**2))*np.sqrt(1- 4*(mF**2/m**2))
            return pre*kin
    
    def calcWidthQuarksDP(self,g):
        mass = []
        widthscc = []
        widthsbb = []
        widthstt = []
        m = 0
        while m < 3.0 :
            mass.append(m)
            width2cc = self.GammaVff(3.,g,m,2*par.ge/3,par.mc_)
            width2bb = self.GammaVff(3.,g,m,-1*par.ge/3,par.mb_)
            width2tt = self.GammaVfff(3.,g,m,2*par.ge/3,par.mt_)
            widthscc.append(width2cc)
            widthsbb.append(width2bb)
            widthstt.append(width2tt)
            m+=0.001
        return (mass,widthscc,widthsbb,widthstt)
  
    
    ### Widths with DM final states ###
    
    # vector -> DM DM
    def Gamma2DM(self,g, mV, mDM,DMtype="No", splitting=0.1): # Vector Boson -> X X (DM)
        if mV<2*mDM: return 0
        #model-independent prefactor for two body decay
        pre = g**2/48./math.pi*mV*(1-4*mDM**2/mV**2)**(1/2)/mV**2
        #model-dependent matrix element
        me = 0.
        if DMtype in ["complex scalar","Majorana","Majorana fermion"]:
            me = mV**2*(1- (4*mDM**2/mV**2))
            # identical particle factor
            if DMtype in ["Majorana","Majorana fermion"]: me*=2.
        elif DMtype in ["Dirac","Dirac fermion"]:
            me = 4*mV**2*(1+2*mDM**2/mV**2)
        elif DMtype=="inelastic":
            mDM2 = mDM*(splitting+1)
            if mV<mDM+mDM2: return 0
            else:
                pre = g**2/48./math.pi*((1-(mDM+mDM2)**2/mV**2)*(1-(mDM*splitting)**2/mV**2))**0.5
                me = 4.*mV*((1+2*mDM*mDM2/mV**2)-(mDM*splitting)**2*(1+(mDM+mDM2)**2/mV**2)/2./mV**2) #fixed mV**2 -> mV
        else:
            print("DM type not specified")
        return pre*me


class DecayError(Exception):
    """
    Error for the 'decayZp' class.
    """
    pass

class decayZp(Processes):
    """
    Calculate the Width and Bfrac of the Dark Boson
    """
    ###########################################################################
    def __init__(self, name, coup, DMtype = "inelastic", Delta =0.4, Rrat= 3 , mhad= 1.73, invchannel = ["neutrinos", "dark"]):

        
        self.modelname = name
        self.__cache = {}
        
        # hadron-quark transition
        self.mhad = mhad
        # mediator-dark matter mass ratio mV/mDM
        self.Rrat = Rrat
        
        # dark matter settings
        # DM type
        if DMtype not in ch.allDMtypes:
            raise ValueError(f"Invalid DM type '{DMtype}'. Allowed types are: {', '.join(ch.allDMtypes)}")
        self.DM = DMtype
        self.split = Delta
        
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

        
        # define the invisible and  viisible states
        self.invstates = invchannel
        self.visstates = ["gamma_gamma_gamma", "e_e", "mu_mu", "tau_tau", "QCD"]
                
        self.gQstates = ["gamma_gamma_gamma", "e_e", "mu_mu", "tau_tau", "QCD", "neutrinos"]
        self.gDstates = ["dark"]
        self.channels = self.gQstates+self.gDstates

    ###########################################################################
    def compHADch(self, m, gQ=1, hadchannels = "total"):
        """
        Hadronic width
        """
        wHtot = 0
        
        if hadchannels== "total":
            # initialization
            for channel in ch.Fchannels:
                channel.resetParameters(1,0.,0.,0.,self.cferm["u"](m),self.cferm["d"](m),self.cferm["s"](m))
            
            for i in range(0, len(ch.Fchannels)):
                width = 0.
                if m < 2.0: width = getattr(ch.Fchannels[i], 'GammaDM')(m)
                width *= gQ**2
                wHtot += width
        
        else:
            for hadch in (hadchannels,) if isinstance(hadchannels, str) else hadchannels:
                func,mode = ch.allchannels.get(hadch)
                func.resetParameters(1,0.,0.,0.,self.cferm["u"](m),self.cferm["d"](m),self.cferm["s"](m))
                
                if mode<0:
                    whad= getattr(func,"GammaDM")(m)
                    wHtot += gQ**2 * whad
                else:
                    whad= getattr(func,"GammaDM_mode")(m,mode)
                    wHtot += gQ**2 * whad
        
        return wHtot
        
    
    ###########################################################################
    def normwidth(self, states, m, gQ = 1.0, gD = 1.0, mDMval = None):
        """
        Calculates the width[GeV] for a given set of states 
        considering the following model parameters:

        states:  final state or states.
        m:       mediator mass Zq (GeV).
        gQ:      global coupling (unitless).
        gD:      dark coupling
        """
        
        ## fixed couplings at m
        cfermfix = {}
        for key, func in self.cferm.items():
            cfermfix[key] = func(m)

        # Loop over the states.
        #total,
        totgQ,totgD = 0, 0
        partgQ, partgD = 0, 0
         
        for state in (states,) if isinstance(states, str) else states:

            # Use cached result if valid.
            cache = self.__cache.get(state)
            if cache and cache[0] == m:
                totgQ += cache[1]
                totgD += cache[2] 
                continue
    
            chel = state.split("_")
            
            if state == "invisible":
                partgQ= self.normwidth([ch for ch in self.invstates  if ch in self.gQstates],m)
                partgD= self.normwidth([ch for ch in self.invstates  if ch in self.gDstates],m)

                # part = self.normwidth(self.invstates, m, gQ, gD)
                
            elif state == "visible":
                partgQ = self.normwidth(self.visstates, m) #, gQ, gD)
            
            elif state == "neutrinos":
                partgQ = self.normwidth(["nue_nue", "numu_numu", "nutau_nutau"], m) #, gQ, gD)
                
            # "QCD" corresponds to hadron before mhad and to quarks after    
            elif state == "QCD":
                if m <= self.mhad:
                    partgQ = self.normwidth(["hadrons"], m)
                if m > self.mhad:
                    partgQ = self.normwidth(["quarks"], m)   
                
            elif state == "hadrons":
                partgQ = self.compHADch(m, hadchannels = "total" )
                
            elif state == "quarks":
                partgQ = self.normwidth(["u_u", "d_d", "s_s", "c_c","b_b", "t_t"], m)           
          
            elif state == "total":
                partgQ = self.normwidth(self.gQstates, m)
                partgD = self.normwidth(self.gDstates, m)
                    
            elif state == "dark":
                if mDMval: mDM = mDMval
                else: mDM = m/self.Rrat
                partgD = self.Gamma2DM(1, m, mDM, DMtype= self.DM, splitting=self.split)
                

            # Perturbative decay into a fermion pair, equation 2.13.
            elif len(chel) == 2 and chel[0] == chel[1] and chel[0] in par.mferm:
                ferm = chel[0]
                
                ff, mf, cf = par.fferm[ferm], par.mferm[ferm], cfermfix[ferm]
                
                partgQ = self.GammaVff(ff, 1, m, cf, mf)      
                
            # Perturbative decay into three photons via an electron loop,
            # equation 3.5 of Seo:2020dtx.
            elif len(chel) == 3 and chel[0] == chel[1] == chel[2] == "gamma":
                mf, xf = par.mferm["e"], cfermfix["e"]
                if m <= 2.0*mf: partgQ = (
                    (xf**2.0*par.ge**6.0)/(4.0*math.pi)**4.0/(
                    2.0**7.0*3.0**6.0*5.0**2.0*math.pi**3.0)*(m**9.0/mf**8.0)*(
                    17.0/5.0 + (67.0*m**2.0)/(42.0*mf**2.0) +
                    (128941.0*m**4.0)/(246960.0*mf**4.0)))
                else: partgQ = 0


            else:
                raise DecayError("Unknown state '%s'." % state)

            # Cache the result.
            totgQ += partgQ
            totgD += partgD
            # total += (gQ**2)*partgQ + (gQ**2)*partgD
            self.__cache[state] = (m, partgQ, partgD)
        
        return (gQ**2)*totgQ + (gD**2)*totgD

   
    ###########################################################################
    def totwidth(self, m, gQ, gD):
        
        gqwid = self.normwidth(self.gQstates, m, gQ, 1)
        gdwid = self.normwidth(self.gDstates, m, 1, gD)
        
        return gqwid+gdwid
        
         
    ###########################################################################
    def tau(self, m, gQ, gD):
        """
        Calculates the lifetime[s] for a given mass and couplings.
        """
     
        return par.hbar/self.totwidth(m, gQ, gD)
 
             
    ###########################################################################
    def ctau(self, m, gQ, gD):
        """
        Calculates the decay length[m] for a given mass and couplings.
        """
     
        return par.cLight*self.tau(m, gQ, gD)
   

    ###########################################################################
    def bfrac(self, states, m, gQ, gD):
        """
        Calculates the branching fraction for a given states, mass and couplings.
        """
        
        chwid = self.normwidth(states, m, gQ, gD)
        totwid = self.totwidth( m, gQ, gD)
        
        if chwid == 0: return 0.0
        if totwid == 0: return 0.0

        return chwid/totwid


                
class Model(decayZp):
    
    def __init__(self, name, coup, DMtype, split=0., Rrat=3, mhad= None, folder = "models"):
        
        # inhert self variables from decayZp
        super().__init__(name, coup, DMtype, split, Rrat, mhad)
      
        # create model folder 
        self.folder = folder
        self.modfolder = usefunc.create_folder(self.folder+"/%s" % (self.modelname))
        #create folder to save plots
        self.plotfold = usefunc.create_folder(self.modfolder+"/plots")

        # define DM label
        if self.split!= 0: self.DMlab = self.DM.replace(" ", "_") + "_DM_delta_" + str(self.split)
        else:  self.DMlab = self.DM.replace(" ", "_") + "_DM"
        # create DM folder 
        self.dmfolder = usefunc.create_folder(self.folder+"/DM_model")

        if DMtype=="inelastic":
            self.widChi2 = winel.decayChi2(coup,Rrat)
            
    
    def calcwid(self, gQ, gDM, mmin=1e-3 ,mmax=10.0, step=10000):
        
        # set couplings
        self.gQ = gQ
        self.gDM = gDM
        
        # compute normalized widths
        self.calcnormwid(mmin,mmax,step)

        # save the interpolated widths 
        wid_dicts = {'wsm': self.wsmN,
                    'wferm': self.wfermN,
                    'whad': self.whadN }

        for widname, widdict in wid_dicts.items():
            setattr(self, widname, {key: interp1d(self.masses,interp_func(self.masses)* self.gQ**2) 
                                    for key, interp_func in widdict.items()})

        self.wdm = {key: interp1d(self.masses,interp_func(self.masses)* self.gDM**2) 
                    for key, interp_func in self.wdmN.items()}

        self.wtot =  interp1d(self.masses, (self.wsm["total"](self.masses) + self.wdm["DM"](self.masses)),
                              kind='linear', fill_value="extrapolate")
 
        
    def calcnormwid(self, mmin=1e-3 ,mmax=10.0, step=10000):    
              
        # create arrays to store the normalized widths
        self.wsmN_arr = {"QCD":[] , "chlep":[], "neutrinos": [] , "total": []}
        self.wdmN_arr = { "DM": []}
        self.wfermN_arr = {}
        self.whadN_arr = {}
        
        for hadch in ch.labelhad:
            self.whadN_arr[hadch] = []       
        self.whadN_arr["tothad"] = []
                    
        for fch in ch.labellep:
            self.wfermN_arr[fch] = []    
        self.wfermN_arr["quarks"] = []
        
        # check if hadronic width files exist
        self.hwidfile = (self.modfolder+"/%s_hadronic_normwid.txt" % (self.modelname))

        if os.path.isfile(self.hwidfile):
            print("Using hadronic width file...")
            self.load_hwid()
        else:
            print("Computing hadronic width file...")
            self.comp_hwid()
            self.save_hwid()
         
        self.comp_fwid(mmin ,mmax, step)
        
        # transform fermion and hadron arrays into interpolated functions
        self.wfermN = usefunc.interp_dict(self.wfermN_arr, self.masses)
        self.whadN = usefunc.interp_dict(self.whadN_arr, self.hadmasses)
        
        # compute hadron-quark transition self.mhad
        self.had2quark(self.mhad)
        if self.mhad==None or self.mhad<=1.5:
            return "intersection mass between the hadronic and the perturbative quark width has to be chosen first"
        
        # save the QCD width QCD = hadrons below the transition and quarks above it
        hadint_mval = self.whadN["tothad"](self.masses)
        quarkint_mval = self.wfermN["quarks"](self.masses)
        self.wsmN_arr["QCD"] = np.where(self.masses <= self.mhad, hadint_mval, quarkint_mval)
        
        # total
        self.wsmN_arr["total"] = np.sum([self.wsmN_arr["QCD"],self.wsmN_arr["chlep"],self.wsmN_arr["neutrinos"]], axis=0)
        
        # transform sm and dm arrays into interpolated functions
        self.wsmN = usefunc.interp_dict(self.wsmN_arr, self.masses)
        self.wdmN = usefunc.interp_dict(self.wdmN_arr, self.masses)
        
        # save all the widths
        self.save_fwid()
        
    
    def load_hwid(self):
        # loading the hadronic wid values
        haddf = pd.read_csv(self.hwidfile, sep='\t')
        haddfch = haddf.drop(haddf.columns[[0]], axis=1)
        self.whadN_arr = haddfch.to_dict(orient='list')                
        self.hadmasses = haddf.iloc[:, 0].values

    def save_hwid(self):
        # saving the hadronic wid values
        self.save_data(self.whadN_arr, self.hadmasses, filename = "hadronic_normwid" )     
    
    def save_fwid(self):
        # saving the non-hadronic wid values
        self.save_data(self.wfermN_arr, self.masses, filename = "fermionic_normwid" )     
        self.save_data(self.wsmN_arr, self.masses, filename = "final_normwid" )     
        if self.DM != "No": self.save_data(self.wdmN_arr, self.masses,
                                           pathname = self.dmfolder+"/"+self.DMlab+"_normwid.txt" )     
         
    def comp_hwid(self,mmin=par.mpi0_ ,mmax=2.0, step=1860):
     
        self.hadmasses = np.linspace(mmin,mmax, step)
        start_time = time.time()
            
        for im in range(0, len(self.hadmasses)):
            m = self.hadmasses[im]
            tothadw = 0.
            
            for i in range(0, len(ch.Fchannels)):
                wch = self.compHADch( m, gQ=1, hadchannels = ch.labelhad[i] )
                self.whadN_arr[ch.labelhad[i]].append(wch)
                tothadw += wch
            
            self.whadN_arr["tothad"].append(tothadw) 
    
            sys.stdout.write('\r')
            sys.stdout.write("processed up to m="+str(round(m,2))+"GeV ("+str(im)+"/"+str(step)+" mass values) in "+str(round(time.time()-start_time,3))+" seconds")
            sys.stdout.flush()
    
    def comp_fwid(self,mmin=1e-3 ,mmax=10.0, step=10000):
     
        self.masses = np.linspace(mmin, mmax, step)

        for im in range(0, len(self.masses)):
            
            m = self.masses[im]

            # leptons
            wltot  =0 
            for i in range(0, len(ch.labellep)):
                wlep = self.normwidth(ch.labellep[i], m)
                self.wfermN_arr[ch.labellep[i]].append(wlep)
                wltot+=wlep
            
            # quarks
            norm = self.GammaVff(1,1,m,1,par.mlep_[1])
            qwidth = self.RInclusive(m)*norm
            self.wfermN_arr["quarks"].append(qwidth)

            # DM
            widdm =  0
            if self.DM !="No":
                widdm = self.normwidth("dark", m)
            self.wdmN_arr["DM"].append(widdm) 

        
        self.wsmN_arr["chlep"] = np.sum([array for key, array in self.wfermN_arr.items() if any(label in key for label in ch.labelchlep)], axis=0)
        self.wsmN_arr["neutrinos"] = np.sum([array for key, array in self.wfermN_arr.items() if any(label in key for label in ch.labelnu)], axis=0)

    def had2quark(self,mhad=None):
        if mhad != None:
            if mhad<1.5:
                print ('Please, choose a transition value above 1.5 GeV.')
            else:
                self.mhad = mhad
        else:
            print ("\n Calculating the hadron-quark transition...")
            intx = np.argwhere(np.diff(np.sign(np.asarray(self.wfermN["quarks"](self.masses)) - np.asarray(self.whadN["tothad"](self.masses))))).flatten()
            mask = self.masses[intx] <= 2.5
            if intx.size >0 and self.masses[intx[mask][-1]]>=1.5:
                self.mhad = self.masses[intx[mask][-1]]
                print ("The transition from hadrons to quarks will happen at "+str(round(self.mhad,3))+" GeV.")
            elif self.masses[intx[-1]]<1.5:
                print ("The value of mass found for the intersection mhad=",round(self.masses[intx[-1]],3),"is inappropriate (below 1.5 GeV). Please set by hand the transition mass.")
            else: 
                print ("The function failed to find a intersection mass between the hadronic and the perturbative quark width. Please set by hand the transition mass.")


    def calcbr(self, gQ, gDM, mmin=1e-3 ,mmax=10.0, step=10000):
        
        self.calcwid(gQ, gDM, mmin=1e-3 ,mmax=10.0, step=10000)

        br_dicts = {'brsm': self.wsm,
                    'brdm': self.wdm,
                    'brferm': self.wferm }

        self.brhad = {key: interp1d(self.hadmasses,  func((self.hadmasses)) / self.wtot(self.hadmasses), kind='linear')  for key, func in self.whad.items()}
        for brchan, widdict in br_dicts.items():
            setattr(self, brchan, {key: interp1d(self.masses,  func((self.masses)) / self.wtot(self.masses), kind='linear')  for key, func in widdict.items()})
     
        
    def save_br(self):
        self.save_data(self.brsm, self.masses, filename = "final_br", stype = "/br" )
        self.save_data(self.brdm, self.masses, filename = "dm_br" , stype = "/br" )
        self.save_data(self.brferm, self.masses, filename = "fermionic_br" , stype = "/br" )
        self.save_data(self.brhad, self.hadmasses, filename = "hadronic_br" , stype = "/br" )

    def save_data(self, input_dict, marr, filename = "save_data" , stype = None, pathname = None):
        # saves the interpolated functions of a dictionary int_dict applied to an array marr into a filename
        if stype:
            sdir = usefunc.create_folder(self.modfolder+stype)
            savename = sdir+"/%s_%s_%s_R_%s_gQ_%s_gDM_%.1f.txt " % (self.modelname, self.DMlab, filename, self.Rrat, self.gQ, self.gDM)
        else: 
            savename = self.modfolder+"/%s_%s.txt" % (self.modelname, filename)
            
        if pathname: savepath = pathname
        else: savepath = (savename)
        
        if any(isinstance(val, list) or isinstance(val, np.ndarray) for val in input_dict.values()):
            data = pd.DataFrame(input_dict)
        if any(isinstance(val, interp1d) for val in input_dict.values()):
            data = pd.DataFrame({key: interp_func(marr) for key, interp_func in input_dict.items()})
        
        data.insert(0, 'mass[GeV]',  marr)
        data.to_csv(savepath, index=False, sep='\t')          
        
      
    #---------------  plot widths -------------------# 
    def plotwid(self,xrange=[0.1,2.],yrange=[1e-9,2.], name=None, Wsingle_had=[],Wsingle_ferm=[], W_DM=True):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        
        ax.plot(self.masses,self.wtot(self.masses), c='black', lw =1.3, label='total')
        
        plot_params = [('QCD', 'orangered', 'QCD'),
                       ('neutrinos', 'forestgreen', 'neutrinos'),
                       ('chlep', 'dodgerblue', 'charged leptons')]

        for key, color, label in plot_params:
            width_array = self.wsm[key](self.masses)
            if not usefunc.is_all_zero(width_array):
                ax.plot(self.masses, width_array, c=color, lw=1.3, label=label)

        for hadch in Wsingle_had:
            ax.plot(self.masses,self.whad[hadch](self.masses), lw =1.2, label=hadch)
                
        for fermch in Wsingle_ferm:
            ax.plot(self.masses,self.wferm[fermch](self.masses), lw =1.2, label=fermch)

        couplab = " $g_Q =$ %s" %(usefunc.sci_not(self.gQ))
        ## DM width ##  
        Wtype= None
        if self.DM != "No":
            Wtype = self.DMlab.replace("_", " ")
            couplab = couplab + "\n $g_{DM} = %.1f$" %(self.gDM)
            if W_DM==True: ax.plot(self.masses,self.wdm["DM"](self.masses),c= "darkviolet",lw=1.2,linestyle="dashdot",label= Wtype)
        elif self.DM=="No": Wtype = 'SM'           
        
        plt.text(xrange[0]*(1.2), yrange[0]*3, " $R=%s$ \n" %(self.Rrat) + couplab ,
                 color="0.2", fontsize = 12, bbox=dict(facecolor='white', edgecolor= 'none', alpha=0.5))
        
        ###############
        ax.set_title(self.modelname.replace("_", " ") + " + " + Wtype, fontsize=16)
        plt.xlabel("$m_{Z_Q}$ [GeV]",fontsize=14)
        plt.ylabel("$\\Gamma \\, (\\; Z_{Q} \\; \\to \\; F )$",fontfamily= 'sans-serif',fontsize=14)
        ax.legend(loc="best", ncol=2)      
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        plt.axvline(self.mhad,lw=1.5,c ='0.3',ls='--',label="transition")
        plt.yscale("log")
        plt.minorticks_on()
        #plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.2)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if name:
            plotlab = self.modelname.replace(" ","_")+"_"+Wtype.replace(" ","_")+"_R_%s_gQ_%s_gDM_%.1f" % (self.Rrat,self.gQ, self.gDM)
            if self.DM == "No": plotlab = plotlab[0:-8]
            plt.savefig(self.plotfold+'/width_'+plotlab+'_'+name+".pdf")
        plt.show()
        plt.close()
        return fig
    

    #---------------  plot brs -------------------# 
    def plotbr(self,xrange=[0.1,2.],yrange=[1.e-3,2.], BRsingle_had=[],BRsingle_ferm=[],BR_DM=True,name=None):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        
        plot_params = [('QCD', 'orangered', 'QCD'),
                       ('neutrinos', 'forestgreen', 'neutrinos'),
                       ('chlep', 'dodgerblue', 'charged leptons')]

        for key, color, label in plot_params:
            br_array = self.brsm[key](self.masses)
            if not usefunc.is_all_zero(br_array):
                ax.plot(self.masses, br_array, c=color, lw=1.3, label=label)

        for hadch in BRsingle_had:
            ax.plot(self.hadmasses,self.brhad[hadch](self.hadmasses), lw =1.2, label=hadch)
                
        for fermch in BRsingle_ferm:
            ax.plot(self.masses,self.brferm[fermch](self.masses), lw =1.2, label=fermch)

        couplab = " $g_Q =$ %s" %(usefunc.sci_not(self.gQ))
        ## DM width ##  
        Wtype= None
        if self.DM != "No":
            Wtype = self.DMlab.replace("_", " ")
            couplab = couplab + "\n $ g_{DM} = %.1f$" %(self.gDM)
            if BR_DM==True: ax.plot(self.masses,self.brdm["DM"](self.masses),c= "darkviolet",lw=1.2,linestyle="dashdot",label= Wtype)
        elif self.DM=="No": Wtype = 'SM'           
        
        plt.text(xrange[0]*(1.2), yrange[0]*3, " $R=%s$ \n" %(self.Rrat) +couplab ,
                 color="0.3", fontsize = 12, bbox=dict(facecolor='white', edgecolor= 'none', alpha=0.5))
        
        ###############
        ax.set_title(self.modelname.replace("_", " ") + " + " + Wtype, fontsize=16)
        plt.xlabel("$m_{Z_Q}$ [GeV]",fontsize=14)
        plt.ylabel("BR ($\\; Z_{Q} \\; \\to \\;$ F )",fontfamily= 'sans-serif',fontsize=14)
        ax.legend(loc= "best", ncol=2)      
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        plt.axvline(self.mhad,lw=1.5,c ='0.3',ls='--',label="transition")
        plt.yscale("log")
        plt.xscale("log")
        plt.minorticks_on()
        #plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.2)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if name:
            plotlab = self.modelname.replace(" ","_")+"_"+Wtype.replace(" ","_")+"_R_%s_gQ_%s_gDM_%.1f" % (self.Rrat,self.gQ, self.gDM)
            if self.DM == "No": plotlab = plotlab[0:-8]
            plt.savefig(self.plotfold+'/br_'+plotlab+'_'+name+".pdf")
        plt.show()
        plt.close()
        return fig    
    
    
    ### functions to compute and plot the normalized Rvalues  ###
    def calcRval(self):

        self.Rhad = {key: interp1d(self.hadmasses, [usefunc.intdiv(func(m),self.GammaVff(1,self.gQ,m,1,par.mferm['mu'])) 
                                                    for m in self.hadmasses], fill_value="extrapolate")
                     for key, func in self.whad.items()}
        
        self.Rquarks = interp1d(self.masses, [self.RInclusive(m) for m in self.masses], fill_value="extrapolate") 
        
    
    def plotRval(self,xrange=[0.3,2.],yrange = None, Rsingle=[], name= None):
        
        fig, ax = plt.subplots(figsize=(8.5, 5))
        
        totRhad = self.Rhad["tothad"](self.hadmasses)
        ax.plot(self.hadmasses,totRhad, c='darkorange', lw =1.3, label='all hadrons')

        ax.plot(self.masses,self.Rquarks(self.masses), c='tab:red', lw =1.2, label='quarks+QCD corrections')
        
        for hadch in Rsingle:
            ax.plot(self.hadmasses,self.Rhad[hadch](self.hadmasses), lw =1.2, label=hadch)
       
        if max(data.RPDG[:880])>min(totRhad)*5:
            ax.errorbar(data.massPDG,data.RPDG, [data.low_eRPDG,data.up_eRPDG] , color ='black', ls ='none', fmt = '.', ms =2.5
                        ,fillstyle='full',elinewidth=0.5, capsize=0.5,capthick = 0.3, label= "PDG data 2020", zorder = -5)
                        
        ax.set_title(self.modelname)
        ax.set_xlabel("$\\sqrt{s}$ [GeV]",fontfamily= 'serif',fontsize=14)
        ax.set_ylabel("$R^{\\mathcal{H}}_{\mu} = \\frac{\\Gamma\;(Z_Q \\; \\to \\; \\mathrm{hadrons})}{\\Gamma\;({Z_Q \\to \\; \\mu^+\\mu^-)}}$",fontfamily= 'serif',fontsize=14)
        ax.legend(loc="best",ncol=2)
       
        ax.set_xlim(xrange)
        if yrange: ax.set_ylim(yrange)
        plt.yscale("log")
        plt.minorticks_on()
        # plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.2)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        if name:
            plotlab = self.modelname.replace(" ","_")+"_R_%s_gQ_%s" % (self.Rrat,self.gQ)
            plt.savefig(self.plotfold+'/rval_'+plotlab+'_'+name+".pdf")      
        plt.show()
        plt.close()
        return fig     


    ### inelastic DM: functions to compute and plot the widths/BRs of the heavier fermion chi2 ###
    def calcwidChi2(self, gQ, gDM, channels = None, mmin=1e-1 ,mmax=3.0, step=1000, save= False):

        # compute chi2 normalized widths
        self.calcnormwidChi2(channels, mmin, mmax, step, save)

        self.wchi2 = {key: interp1d(self.DMmasses,interp_func(self.DMmasses)* (gQ*gDM)**2) 
                      for key, interp_func in self.wchi2N.items()}


                                                   
    def calcnormwidChi2(self, channels = None, mmin=1e-1 ,mmax=3.0, step=1000, save= False):    
              
        # create arrays to store the normalized widths
        self.wchi2N_arr = {}
        
        if channels: states = channels
        else: states = self.widChi2.wc2states
        
        # print (states)
        for chan in states:
            self.wchi2N_arr[chan] = []       

        # m1/DM masses                    
        self.DMmasses = np.linspace(mmin, mmax, step)


        # check if width files exist
        self.c2widfile = (self.modfolder+"/%s_chi2_delta_%s_normwid.txt" % (self.modelname, self.split))

        if os.path.isfile(self.c2widfile):
            print("Using width file...")            
            wc2df = pd.read_csv(self.c2widfile, sep='\t')
            wc2dfch = wc2df.drop(wc2df.columns[[0]], axis=1)
            self.wchi2N_arr = wc2dfch.to_dict(orient='list')                
            self.DMmasses = wc2df.iloc[:, 0].values
            
        
        else:
            print("Computing width file...")
                 
            for chan in states:
                wch = []
                for i in range(len(self.DMmasses)):
                    width = self.widChi2.width(states= chan, m1=self.DMmasses[i], Delta=self.split, gChi=1, gQ=1)
                    wch.append(width)
            
                self.wchi2N_arr[chan] = wch
                
            self.wchi2N_arr["total"] = [sum(elements) for elements in zip(*self.wchi2N_arr.values())]
                        
            if save:
                self.save_data(self.wchi2N_arr, self.DMmasses, filename = "chi2_delta_%s_normwid"  % (self.split))     
            
       
        self.wchi2N = usefunc.interp_dict(self.wchi2N_arr, self.DMmasses)
        

     
    def plotNwidChi2(self,xrange=[0.1,2.],yrange=None, name=None):
        
        fig, ax = plt.subplots(figsize=(8.5, 5))
        
        ax.plot(self.DMmasses,self.wchi2N["total"](self.DMmasses), c='black', lw =1.3, label='total')
        
        
        for chan,wch in self.wchi2N.items():
            plt.plot(self.DMmasses, self.wchi2N[chan](self.DMmasses), ls= "-",lw=2, label = chan)
        

        # plt.text(0.2,0.8, " $R=%s$ \n" %(self.Rrat) , transform=ax.transAxes,
        #          color="0.2", fontsize = 12, bbox=dict(facecolor='white', edgecolor= 'none', alpha=0.5))
        
        ###############
        ax.set_title(self.modelname.replace("_", " "), fontsize=16)
        plt.xlabel("$m_{1}$ [GeV]",fontsize=14)
        plt.ylabel("$ \\Gamma \\; (\chi_2 \\to \chi_1 + {\\rm SM})$",fontfamily= 'sans-serif',fontsize=14)
        ax.legend(title= "$ \\Delta = %s \\, , \\, R=%s$ \n" %(self.split,self.Rrat) ,loc="best", ncol=2)      
        ax.set_xlim(xrange)
        if yrange: ax.set_ylim(yrange)
        plt.yscale("log")
        plt.minorticks_on()
        #plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.2)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if name:
            plotlab = self.modelname.replace(" ","_")+"chi2wid_R_%s" % (self.Rrat)
            if self.DM == "No": plotlab = plotlab[0:-8]
            plt.savefig(self.plotfold+'/width_'+plotlab+'_'+name+".pdf")
        plt.show()
        plt.close()
        return fig
