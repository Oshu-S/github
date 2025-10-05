from matrix import*  
from solver import* 
from pedestrian import* 
from matplotlib import pyplot as plt
import timeit
import numpy as np


#step 1 setup beam and pedestrians
#beam
numElements = 10  # n - Number of beam elements !not for modal
length = 50  # L - Length (m)
width = 2  # b - Width (m)
height = 0.6  # h - Height (m)
E = 200e9  # E - Young's modulus (N/m^2)
modalDampingRatio = 0.005  # xi - Modal damping ratio of the beam
nHigh = 3  # nHigh - Higher mode for damping matrix
beamFreq = 2 #Hz
area = 0.3162  # A - Cross-section area (m^2)
linearMass = 500  # m - Linear mass (kg/m)
x_interested= length/2
numbers = 3
#ped
numped = 1
pedmass = 70.3     #kg
peddamp = .3    
#pedstiff = 25000 #N/m
pedpace  = 2     #Hz
pedphase = 0
pedInlocation = 0
pedvelocity = 1.25
pedBodyF= 2 #Hz

#ped
kped=(2*np.pi*pedBodyF)**2*pedmass
cped = (2*np.pi*pedBodyF)*2*peddamp*pedmass

#mat=np.zeros(1,numped) #to be extended into probabilistic inputs
mped=np.array([pedmass])
cped = np.array([cped])
kped = np.array([kped])

#bridge
modulus =linearMass * ((2 * math.pi * beamFreq) * (math.pi / length) ** (-2)) ** 2  #E*(width*height**3)/12

#set time info
hht=0.01

#initial possition vector.......formultiple ped all these would become matrices

#xrb=np.zeros(1,numped)
xrb=[0]

Bridge = bridge(   
    length = length,                 # m
    modulus = modulus,               # N m^2
    density = linearMass,            # kg/m
    damp    = modalDampingRatio ,    #%
    numbers = 3,  )                   #modes


Human = Pedestrian(
         mass = pedmass,     #kg
         damp = peddamp ,   #%
         stiff = kped, #N/m
         pace  = pedpace ,    #Hz
         phase = pedphase,
         location = pedInlocation,
         velocity = pedvelocity,
         
         iSync=0)

u,du,ddu_hsi = Newmarksuper_HSI (Human,Bridge,numped,numbers,length,hht,pedvelocity,mped,kped,cped,xrb,linearMass)
                
accn_hsi = accdyn_super(Bridge,ddu_hsi,x_interested,hht)
#vertical_displacement = accdyn_super(Bridge,u,25,hht)

# --- FFT of "with HSI" acceleration (accn_hsi) ---
import numpy as np
import matplotlib.pyplot as plt

# Sampling info
dt = hht                     # your time step (0.01 s)
fs = 1.0 / dt
n  = len(accn_hsi)

# (Recommended) remove DC before FFT
x = accn_hsi - np.mean(accn_hsi)

# One-sided spectrum (rfft) → frequencies 0..fs/2
freqs = np.fft.rfftfreq(n, d=dt)
X = np.fft.rfft(x)

# Amplitude spectrum (m/s^2) — normalized by n
amp = np.abs(X) / n

# Optional: convert to amplitude spectral density (per √Hz)
# asd = amp / np.sqrt(fs/2)   # comment out if you just want 'amp'

# Plot
plt.figure(figsize=(9,4.5))
plt.plot(freqs, amp, lw=1.2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s²)")
plt.title("FFT Amplitude Spectrum — With HSI")
plt.xlim(0, fs/2)           # show up to Nyquist
plt.grid(True, ls="--", lw=0.5)
plt.tight_layout()

u,du,ddu = Newmarksuper_HSI (Human,Bridge,numped,numbers,length,hht,pedvelocity,mped,[0],[0],xrb,linearMass)
                
accn = accdyn_super(Bridge,ddu,x_interested,hht)

t = np.arange(0, (length+1) / pedvelocity, hht)
plt.figure(figsize=(9,4.5))
plt.plot(t,accn , label ="without HSI" ,color='r')
plt.plot(t,accn_hsi,label ="with HSI",color='b')
plt.title("mid span acceleration")
plt.xlabel("time(s)")
plt.ylabel("m/s2")
plt.legend()
#plt.plot(t,vertical_displacement)  
plt.show()
#print("ddu",ddu)
#print("accn",accn)


# Create a DataFrame for exporting
import pandas as pd
df = pd.DataFrame({
    "Time (s)": t,
    "Acceleration Without HSI (m/s²)": accn,
    "Acceleration With HSI (m/s²)": accn_hsi
})

# Save to CSV
df.to_csv(r"C:\Users\Admin\OneDrive\Documents\Monash stuff\Year 5\FYP\github\pyhsi_results.csv", index=False)

print("✅ Results saved to 'pyhsi_results.csv'")
