import math

# unit conversion
MeV = 1e-3
gev2nb = 389379.3656

######################
### SM parameters ####
######################

########### parameters ###########
# Constants.
cL    = 2.99792458e8     # Speed of light (m/s)
hbar = 6.58211951e-25   # Reduced Planck's constant (s.GeV)
hvev = 246 # Higgs VEV (GeV)
mhiggs = 125 # Higgs mass (GeV)
alphaEM = 1./137

mpi = 0.1349768 # pion mass (GeV)
fpi = 130e-3 # pion decay constant (GeV)
mK0=0.497611
mKp=0.493677


# lepton masses - PDG values #
me_ = 0.5109989461*MeV #
mmu_ = 0.1056583745
mtau_ = 1.77686
mlep_ = [me_,mmu_,mtau_,0.,0.,0.]
# quark masses #
md_ = 4.67*MeV
mu_ = 2.16*MeV
ms_ = 93.*MeV
mc_ =   1.27
mb_ = 4.18
mt_ = 172.76

# fermion mass
mferm = {
    "e":      5.110e-04,
    "mu":     0.1057,
    "tau":    1.777,
    "nue":    0,
    "numu":   0,
    "nutau":  0,
    "d":      4.67e-03,
    "u":      2.16e-03,
    "s":      93e-03,
    "c":      1.42,
    "b":      4.7,
    "t":      174.3
    }

# coefficients to decay width into leptons (neutrinos have 1/2 since they are always LH, and quarks have the color factor)
fferm = {
    "e":      1.0,
    "mu":     1.0,
    "tau":    1.0,
    "nue":    1.0/2.0,
    "numu":   1.0/2.0,
    "nutau":  1.0/2.0,
    "d":      3.0,
    "u":      3.0,
    "s":      3.0,
    "c":      3.0,
    "b":      3.0,
    "t":      3.0
    }




############
# mesons ###
############
# masses and widths from PDG

## light mesons ##
#pion
mpi0_ = 134.9768*MeV
mpi_  = 139.57039*MeV
fpi = 130e-3 # pion decay constant (GeV)

# rho mass
mRho  = .7755
gRho  = .1494
mRho1 =1.459
gRho1 =0.4
mRho2 =1.72
gRho2 =0.25
# omega mass
mOmega=.78265
gOmega=.00849
# a1 mass
ma1=1.23
ga1=.2
# f0 mass
mf0=1.35
gf0=0.2

# kaon mass
mK0=0.497611
mKp=0.49367

#relativistic degrees of freedoms, we assume g*s=g*

#lightest charmed meson mass
mD0 = 1.864
#lightest bottom meson mass
mUps = 9.46

# fundamental constants #
hbar = 6.58211951e-25   # Reduced Planck's constant (s.GeV)
cLight = 2.99792458e8     # Speed of light (m/s)
ge   = 3.02822e-1  # Electromagnetic coupling (unitless).
hvev = 246 # Higgs VEV (GeV)
mhiggs = 125 # Higgs mass (GeV)
alphaEM = 1./137

# COSMO #
mPlanck = 1.22*10**19 # in GeV
entropy0 = 2891.2 # entropy density in cm-3
rho_c = 1.053672*10**(-5)# critical energy densityin units of h^2 GeV cm-3

#conversion
sigv2cms = 1.1677e-17 # GeV-2 to cm^3/s





