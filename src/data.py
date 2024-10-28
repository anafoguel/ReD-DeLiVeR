import os
import numpy

# data
#---------------  PDG - total rate to hadrons (2020) -------------------# 
dataPDG = numpy.loadtxt(os.path.dirname(os.path.abspath(__file__))+'/data/R_Gamma_PDG_2020.txt')
massPDG = dataPDG[:,0]
RPDG= dataPDG[:,3]
up_eRPDG= dataPDG[:,4]
low_eRPDG= dataPDG[:,5]