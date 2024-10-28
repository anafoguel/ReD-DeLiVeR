import math,cmath
from scipy.interpolate import interp1d
from scipy import integrate
import numpy as np
import F3pi,alpha,Resonance
import yoda,os
import matplotlib.pyplot as plt

mPi_=0.140

# Q02 expressed in other variables
def QZero2(s,Qp2,Qm2):
    return 3*mPi_**2+s-Qm2-Qp2

# bounds for Qm2
def bounds_Qm(s):
    upp = (math.sqrt(s)-mPi_)**2
    low = 4.*mPi_**2
    return (low, upp)

# bounds for Qp2
def bounds_Qp(Qm2,s):
    E2s = 0.5*math.sqrt(Qm2)
    E3s = 0.5*(s-Qm2-mPi_**2)/math.sqrt(Qm2)
    low = (E2s+E3s)**2-(math.sqrt(E2s**2-mPi_**2)+math.sqrt(E3s**2-mPi_**2))**2
    upp = (E2s+E3s)**2-(math.sqrt(E2s**2-mPi_**2)-math.sqrt(E3s**2-mPi_**2))**2
    return (low,upp)

i = 0
j = 0
def F2(Qp2,Qm2,s):
    Q02 = F3pi.QZero2(s,Qp2,Qm2)
    Lorentzpart = F3pi.Lcontracted(Qp2,Qm2,Q02,s)
    formfactor = 0.
    if i==0: formfactor=F3pi.F1(s,Q02)
    elif i==1: formfactor=F3pi.F0_Omega(s,Qp2,Qm2,Q02)
    elif i==2: formfactor=F3pi.F0_Phi(s,Qp2,Qm2,Q02)
    
    if j==0: formfactor*=np.conj(F3pi.F1(s,Q02))
    elif j==1: formfactor*=np.conj(F3pi.F0_Omega(s,Qp2,Qm2,Q02))
    elif j==2: formfactor*=np.conj(F3pi.F0_Phi(s,Qp2,Qm2,Q02))
    return Lorentzpart*formfactor

def complex_quadrature(func, ranges, arguments):
    def real_func(x,y,z):
        return scipy.real(func(x,y,z))
    def imag_func(x,y,z):
        return scipy.imag(func(x,y,z))
    real_integral = integrate.nquad(real_func, ranges, args=arguments)[0]
    imag_integral = integrate.nquad(imag_func, ranges, args=arguments)[0]
    return real_integral + complex(0,1)*imag_integral

def coefficients(s):
    Fcoeff  = np.zeros((3,3),dtype=complex)
    for ix in range(0,3):
        for jx in range(0,3):
            global i,j
            i=ix
            j=jx
            #Fcoeff[ix,jx] = complex_quadrature(F2, [F3pi.bounds_Qp,F3pi.bounds_Qm], args=(s,))[0]
            Fcoeff[ix,jx] = complex_quadrature(F2, [F3pi.bounds_Qp,F3pi.bounds_Qm], (s,))
    return Fcoeff

#coeff = []
#energies = []
#scale = 3.*F3pi.mPi_+0.001
#while scale < 4.0 :
#    print scale
#    s = scale**2
#    energies.append(scale)
#    coeff.append(coefficients(s))
#    if(scale<=1.1) :
#        scale+=0.001
#    else :
#        scale+=0.01

#np.save("3pi_coefficients.npy",[energies,coeff])

[energies, coefficients] = np.load("3pi/3pi_coefficients.npy",allow_pickle=True)

coeffs = {}
for xen in range(0,len(energies)):
    coeffs[energies[xen]] = coefficients[xen]

# coupling strength to rho, phi contributions


cRho_ = 1.
cOmega_ = 1.
cPhi_ = 1.

cVector = [cRho_,cOmega_,cPhi_]

def hadronic_current(energy):
    s=energy**2
    pre= 1./(2*math.pi)**3/32./s**2
    total=0
    for ix in range(0,3):
        for jx in range(0,3):
            total+=pre*cVector[ix]*cVector[jx]*coeffs[energy][ix,jx]
    return total

hadronic_interpolator = None
def readHadronic_Current():
    x = []
    y = []
    for energy in energies:
        x.append(energy)
        y.append(hadronic_current(energy))
    global hadronic_interpolator
    hadronic_interpolator = interp1d(x, y, kind='cubic',fill_value="extrapolate")

def sigmaSM(s):
    en = math.sqrt(s)
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    had = abs(hadronic_interpolator(en))
    return pre*had*Resonance.gev2nb

readHadronic_Current()

xSM = []
ySM = []
en = 0.5
while en<4.0:
    s=en**2
    xSM.append(en)
    #value = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s*Resonance.gev2nb
    #value*= 1./(2*math.pi)**3/32./s**2*Resonance.gev2nb
    #value*=abs(hadronic_interpolator(en))
    value=sigmaSM(s)
    ySM.append(value)
    en+=0.01

path="../xsec_files/data_files/"
# data
analyses={"BABAR_2004_I656680" : ["d01-x01-y01"],
          "SND_2003_I619011" : ["d01-x01-y01"],
          "SND_2002_I582183" : ["d01-x01-y01"]
}

x_data=[]
y_data=[]
e_data=[]
for analysis in analyses :
    aos=yoda.read(os.path.join(os.getcwd(),path)+analysis+".yoda")
    for plot in analyses[analysis] :
        histo = aos["/REF/%s/%s" %(analysis,plot)]
        for point in histo.points() :
            if(analysis == "SND_2003_I619011" or analysis=="SND_2002_I582183"):
                xx = point.x()/1000.
            else:
                xx = point.x()
            x_data.append(xx)
            y_data.append(point.y())
            e_data.append(point.yErrAvg())

    
plt.plot(xSM,ySM,color="blue",label="SM")
plt.errorbar(x_data, y_data, e_data,color="black",linestyle="None")
plt.legend()
plt.yscale("log")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
#plt.show()
plt.savefig("test_3pions.pdf")
#print energies
#print coefficients
