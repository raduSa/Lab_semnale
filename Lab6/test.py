import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, freqz

# Filter parameters
fs = 1000       # Sampling frequency (Hz)
fc = 100        # Cutoff frequency (Hz)
order = 4       # Filter order
rp = 1          # Passband ripple for Chebyshev I (dB)

# Normalized cutoff frequency
Wn = fc / (fs/2)

# Design Butterworth filter
b_butt, a_butt = butter(order, Wn, btype='low')

# Design Chebyshev Type I filter
b_cheb, a_cheb = cheby1(order, rp, Wn, btype='low')

# Frequency response
w_butt, h_butt = freqz(b_butt, a_butt, worN=8000)
w_cheb, h_cheb = freqz(b_cheb, a_cheb, worN=8000)

# Convert to Hz
freq_butt = w_butt * fs / (2*np.pi)
freq_cheb = w_cheb * fs / (2*np.pi)

# Plot magnitude responses
plt.figure(figsize=(10,6))
plt.plot(freq_butt, 20*np.log10(abs(h_butt)), label='Butterworth', linewidth=2)
plt.plot(freq_cheb, 20*np.log10(abs(h_cheb)), label='Chebyshev I', linewidth=2)
plt.axvline(fc, color='k', linestyle='--', label='Cutoff frequency')
plt.title('Frequency Response: Butterworth vs Chebyshev I')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.ylim([-60, 5])
plt.grid(True)
plt.legend()
plt.show()
