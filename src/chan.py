import math
from src.form_factors import F2pi,F3pi,F4pi,F6pi,FPiGamma,FEtaGamma,FK,FKKpi,FKKpipi,FEtaOmega,FEtaPhi,FPhiPi,FPhiPiPi,FOmegaPion,FOmPiPi,FEtaPiPi,FEtaPrimePiPi,Fppbar

######################
### decay channels ###
######################

## hadronic channels ##

# form factors #
Fchannels = [F2pi,F3pi,FPiGamma,FEtaGamma,FEtaOmega,FEtaPhi,FPhiPi,FOmegaPion,FEtaPiPi,FEtaPrimePiPi,F4pi,F6pi,FK,FKKpi,FKKpipi,FPhiPiPi,FOmPiPi,Fppbar]    
# label #
labelhad = ["2pi","3pi","PiGamma","EtaGamma","EtaOmega","EtaPhi","PhiPi","OmegaPion","EtaPiPi","EtaPrimePiPi",
            "4pi","6pi","KK","KKpi","KKpipi","PhiPiPi","OmPiPi","NNbar"]

## leptonic channels ##

# labels #
labelchlep = ["e_e","mu_mu", "tau_tau"]
labelnu = ["nue_nue","numu_numu","nutau_nutau"]
labellep = labelchlep + labelnu
labelquarks = ["u_u", "d_d", "s_s", "c_c","b_b", "t_t"]
labelferm = labellep + labelquarks

## quarks channels (up, down, strange, charm, bottom, top) ##
labelpert= ["quarks"]

# labels combined #
label = labelhad+labellep+labelpert

allDMtypes = ["complex scalar","Majorana","Majorana fermion","Dirac","Dirac fermion","inelastic","No"]

# all possibilities
allchannels = {'2pi':[F2pi,-1],
               '3pi':[F3pi,-1] ,
               'PiGamma':[FPiGamma,-1],
               'EtaGamma':[FEtaGamma,-1],
               'EtaOmega':[FEtaOmega,-1],
               'EtaPhi':[FEtaPhi,-1],
               'PhiPi':[FPhiPi,-1],
               'OmegaPion':[FOmegaPion,-1],
               'EtaPiPi':[FEtaPiPi,-1],
               'EtaPrimePiPi':[FEtaPrimePiPi,-1],
               '6pi':[F6pi,-1],'6pi_c':[F6pi,1],'6pi_n':[F6pi,0],
               'KK':[FK,-1],'KK_c':[FK,1],'KK_n':[FK,0],
               'KKpi':[FKKpi,-1],'KKpi_0':[FKKpi,0], 'KKpi_1':[FKKpi,1],'KKpi_2':[FKKpi,2],
               'KKpipi':[FKKpipi,-1],'KKpipi_0':[FKKpipi,0],'KKpipi_1':[FKKpipi,1],'KKpipi_2':[FKKpipi,2],'KKpipi_3':[FKKpipi,3],
               'PhiPiPi':[FPhiPiPi,-1],'PhiPiPi_n':[FPhiPiPi,0],'PhiPiPi_c':[FPhiPiPi,1],
               'OmPiPi':[FOmPiPi,-1],'OmPiPi_n':[FOmPiPi,0],'OmPiPi_c':[FOmPiPi,1],
               'NNbar':[Fppbar,-1],'ppbar':[Fppbar,0],'nnbar':[Fppbar,1],
               '4pi':[F4pi,-1],'4pi_c':[F4pi,1],'4pi_n':[F4pi,0]}

