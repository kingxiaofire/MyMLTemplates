import numpy as np 
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq

#  sample points 
N = 1200 
 
# sample spacing 
T = 1.0 / 1600.0 
 
x = np.linspace(0.0, N*T, N, endpoint=False) 
sum = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x) 
 
plt.plot(sum) 
plt.title('Sine wave') 
plt.xlabel('Time') 
plt.ylabel('Amplitude') 
plt.grid(True, which='both') 
plt.show()