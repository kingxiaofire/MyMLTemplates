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
plt.savefig('./Code/Test.png')
plt.show()
plt.cla()


sumf = fft(sum) 
xf = fftfreq(N, T)[:N//2] 
plt.ylabel('frequency') 
plt.xlabel('sample') 
plt.title("FFT of sum of two sines") 
plt.plot(xf, 2.0/N * np.abs(sumf[0:N//2])) 
plt.savefig('./Code/Test1.png')
plt.cla()

