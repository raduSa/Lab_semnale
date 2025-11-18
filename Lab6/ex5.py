import numpy as np
import matplotlib.pyplot as plt

square = lambda N: np.ones(N)
hanning = lambda N: 0.5 * (1 - np.cos(2*np.pi*np.arange(N) / N))
signal = lambda t: np.sin(200 * np.pi * t)
Nw = 200

fig, axs = plt.subplots(3)

x = np.linspace(0, 0.2, Nw)

axs[0].plot(x, np.vectorize(signal)(x))
axs[0].set_title('Original')
axs[1].plot(x, square(Nw) * np.vectorize(signal)(x))
axs[1].set_title('Square')
axs[2].plot(x, hanning(Nw) * np.vectorize(signal)(x))
axs[2].set_title('Hanning')

plt.tight_layout()    
plt.savefig(fname=f'Lab6/ex5_fig1.pdf')
plt.show()