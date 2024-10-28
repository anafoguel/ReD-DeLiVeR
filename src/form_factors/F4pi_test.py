# Libraries to load
import math,random,numpy,Resonance,alpha,glob
from scipy.interpolate import interp1d
import F4pi,yoda
import matplotlib.pyplot as plt
import os

hadronic_interpolator_n = None
hadronic_interpolator_c = None

def readHadronic_Current():
    global hadronic_interpolator_n
    global hadronic_interpolator_c
    F4pi.readCoefficients()
    # neutral: pi+pi-2pi0
    x = []
    y = []
    for (key,val) in sorted(F4pi.coeffs_neutral.iteritems()) :
        en = key
        s=en**2
        x.append(en)
        (npoints,wgt,wgt2) = val
        hadcurr,hadcurr_err = F4pi.hadronic_current(s,npoints,wgt,wgt2,omegaOnly=False)
        y.append(hadcurr)
    hadronic_interpolator_n = interp1d(x, y, kind='cubic',fill_value="extrapolate")
    # charged: 2pi+2pi-
    x = []
    y = []
    for (key,val) in sorted(F4pi.coeffs_charged.iteritems()) :
        en = key
        s=en**2
        x.append(en)
        (npoints,wgt,wgt2) = val
        hadcurr,hadcurr_err = F4pi.hadronic_current(s,npoints,wgt,wgt2,omegaOnly=False)
        y.append(hadcurr)
    hadronic_interpolator_c = interp1d(x, y, kind='cubic',fill_value="extrapolate")

def sigmaSM(s,mode) :
    # leptonic part contracted
    sqrts = math.sqrt(s)
    pre = 16.*math.pi**2*alpha.alphaEM(s)**2/3./s
    #prefactor of phase space
    pre *=(2.*math.pi)**4/2./s*Resonance.gev2nb
    if mode=="neutral": hadcurr = hadronic_interpolator_n(sqrts)
    if mode=="charged": hadcurr = hadronic_interpolator_c(sqrts)    
    return pre*hadcurr

readHadronic_Current()

low_lim = 0.62
upp_lim = 4.0

xSM = []
ySM_n = []
ySM_c = []

scale=low_lim
while scale<upp_lim:
    s=scale**2
    xSM.append(scale)
    ySM_n.append(sigmaSM(s,"neutral"))
    ySM_c.append(sigmaSM(s,"charged"))
    scale+=0.01


analyses= { "neutral" : {}, "charged" : {}}
path="../xsec_files/data_files/"
# Loading some data files to compare fit function and data
analyses["neutral"]["BABAR_2017_I1621593"]  = ["d01-x01-y01"]
analyses["charged"]["BABAR_2012_I1086164"] = ["d01-x01-y01"]

br_omega_pi_gamma = 0.084

# function to read analysis
pb=["BESII_2008_I801210","BESII_2007_I750713"]
def readAnalysis(analysis,plots,x,y,e,pre=1.) :
    aos=yoda.read(os.path.join(os.path.join(os.getcwd(),path),analysis+".yoda"))
    fact = 1.
    if(analysis in pb) : fact = 0.001
    for plot in plots :
        histo = aos["/REF/%s/%s" %(analysis,plot)]
        for point in histo.points() :
            xp = point.x()
            yp = point.y()*fact*pre
            ep = point.yErrAvg()*fact*pre
            if(xp>100.) : xp *=0.001
            sys=0.
            if(analysis=="BABAR_2017_I1621593") :
                if plot == "d01-x01-y01" :
                    if(xp<1.2) :
                        sys=(0.455*xp- 0.296)
                    elif(xp<2.7) :
                        sys = 0.031*yp
                    elif(xp<3.2):
                        sys=0.067*yp
                    else :
                        sys=0.072*yp
                else :
                    sys=0.1*yp
            elif(analysis=="BABAR_2012_I1086164") :
                if(xp<1.1) :
                    sys = 0.107*yp
                elif(xp<2.8) :
                    sys = 0.024*yp
                elif(xp<4.) :
                    sys = 0.055*yp
                else :
                    sys = 0.085*yp
            ep = ep+sys
            #ep = math.sqrt(ep**2+sys**2)
            x.append(xp)
            y.append(yp)
            e.append(ep)

# 2pi+2pi- data
x_charged=[]
y_charged=[]
e_charged=[]
for analysis in analyses["charged"] :
    readAnalysis(analysis,analyses["charged"][analysis],x_charged,y_charged,e_charged)

# pi+pi-2pi0 data
x_neutral=[]
y_neutral=[]
e_neutral=[]
for analysis in analyses["neutral"] :
    readAnalysis(analysis,analyses["neutral"][analysis],x_neutral,y_neutral,e_neutral)
    
# neutral
plt.plot(xSM,ySM_n,color="blue",label="SM")
plt.yscale("log")
plt.errorbar(x_neutral, y_neutral, e_neutral,color="black",linestyle="None",label="data")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.xlim(0.7,3.0)
plt.title("$e^+e^- \\to \\pi^+\\pi^- 2\\pi^0$")
plt.legend()
plt.savefig("test_4pi_neutral.pdf")
plt.clf()
plt.cla()
plt.close()

# charged
plt.plot(xSM,ySM_c,color="blue",label="SM")
plt.yscale("log")
plt.errorbar(x_charged, y_charged, e_charged,color="black",linestyle="None",label="data")
plt.xlabel("$\\sqrt{s}$/GeV")
plt.ylabel("$\\sigma$/nb")
plt.xlim(0.5,3.0)
plt.ylim(0.005,100)
plt.title("$e^+e^- \\to 2\\pi^+2\\pi^-$")
plt.legend()
plt.savefig("test_4pi_charged.pdf")
plt.clf()
plt.cla()
                
    
