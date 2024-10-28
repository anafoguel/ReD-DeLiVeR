import Resonance,alpha,cmath,math
import FKKpi,FPhiPi
from scipy.interpolate import interp1d
import numpy as np
import yoda,os
import matplotlib.pyplot as plt

g2=5.37392360229
br00 =.342
brpp =.489

#low_lim = 1.21
#upp_lim = 4.0
#for imode in range(0,3):
#    coeff = []
#    energies = []
#    scale = low_lim
#    print "####### in mode: ", imode, " ##################"
#    while scale < upp_lim :
#        print scale
#        s = scale**2
#        energies.append(scale)
#        coeff.append(FKKpi.calculateIntegrals(s,imode))
#        scale+=0.005
#
#    np.save("KKpi_coefficients_%d.npy" %imode,[energies,coeff])

hadronic_interpolator_0 = None
hadronic_interpolator_1 = None
hadronic_interpolator_2 = None
def readHadronic_Current():
    for imode in range(0,3):
        [energies, integral_values] = np.load("KKpi/KKpi_coefficients_%d.npy" %imode,allow_pickle=True)
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
            A0,A1 = FKKpi.isoSpinAmplitudes(s)
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
readHadronic_Current()

def sigmaSM(s,imode):
    en = math.sqrt(s)
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    pre*= 1./math.sqrt(s)
    if imode==0: had = abs(hadronic_interpolator_0(en))
    if imode==1: had = abs(hadronic_interpolator_1(en))
    if imode==2: had = abs(hadronic_interpolator_2(en))
    return pre*had*Resonance.gev2nb



xSM = []
ySM_0 = []
ySM_1 = []
ySM_2 = []
en = 0.5
while en<4.0:
    s=en**2
    xSM.append(en)
    #value = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s*Resonance.gev2nb
    #value*= 1./(2*math.pi)**3/32./s**2*Resonance.gev2nb
    #value*=abs(hadronic_interpolator(en))
    ySM_0.append(sigmaSM(s,0))
    ySM_1.append(sigmaSM(s,1)+br00*FPhiPi.sigmaSMPhiPi(s))
    ySM_2.append(sigmaSM(s,2)+brpp*FPhiPi.sigmaSMPhiPi(s))
    en+=0.01

def readData() :
    x_out=[]
    y_out=[]
    e_out=[]
    yNorm_out=[]
    eNorm_out=[]
    for analysis in analyses :
        aos=yoda.read(os.path.join(os.getcwd(),path)+analysis+".yoda")
        for plot in analyses[analysis] :
            histo = aos["/REF/%s/%s" %(analysis,plot)]
            for point in histo.points() :
                x = point.x()
                if(x>100.) : x *=0.001
                if(x>3.) : continue
                x_out.append(x)
                y_out.append(point.y())
                e_out.append(point.yErrAvg())
                yNorm_out.append(point.y())
                eNorm_out.append(point.yErrAvg())
    return (x_out,y_out,e_out,yNorm_out,eNorm_out)

path="../xsec_files/data_files/"
# load K0K0pi0 data
analyses={}
analyses["BABAR_2017_I1511276"] = ["d01-x01-y01"]
analyses["SND_2018_I1637194"]   = ["d01-x01-y01"]
x_0,y_0,e_0,yNorm_0,eNorm_0 = readData()
# load KpKmpi0 data
analyses={}
analyses["BABAR_2008_I765258"] = ["d02-x01-y01"]
#analyses["DM2_1991_I318558"  ] = ["d02-x01-y01"]
x_1,y_1,e_1,yNorm_1,eNorm_1 = readData()
# load KpK0pim data
analyses={}
analyses["BABAR_2008_I765258"] = ["d01-x01-y01"]
#analyses["DM1_1982_I176801"  ] = ["d01-x01-y01"]
#analyses["DM2_1991_I318558"]   = ["d01-x01-y01"]
x_2,y_2,e_2,yNorm_2,eNorm_2 = readData()

plt.plot(xSM,ySM_0,color="blue",label="SM")
plt.errorbar(x_0, y_0, e_0,color="black",linestyle="None")
plt.legend()
plt.yscale("log")
plt.xlim(1.2,2.2)
plt.title("$e^+e^- \\to K^0_SK^0_L\\pi^0$")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.savefig("test_KKpi_K0K0Pi0.pdf")
plt.close()


plt.plot(xSM,ySM_1,color="blue",label="SM")
plt.errorbar(x_1, y_1, e_1,color="black",linestyle="None")
plt.legend()
plt.yscale("log")
plt.xlim(1.2,2.2)
plt.title("$e^+e^- \\to K^+K^-\\pi^0$")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.savefig("test_KKpi_KpKmPi0.pdf")
plt.close()

plt.plot(xSM,ySM_2,color="blue",label="SM")
plt.errorbar(x_2, y_2, e_2,color="black",linestyle="None")
plt.legend()
plt.yscale("log")
plt.xlim(1.2,2.2)
plt.title("$e^+e^- \\to K^\\pm K^0_S\\pi^\\mp$")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.savefig("test_KKpi_KpmK0pimp.pdf")
plt.close()
