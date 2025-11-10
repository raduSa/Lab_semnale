import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

x = np.genfromtxt('Lab5/archive/Train.csv', delimiter=',')
x = x[1:]

N = x.shape[0]
x = x[:, 2]
X = np.fft.fft(x)
X_abs = np.abs(X)

# graph k highest frequencies
k_values = [2, 10, 50, 1000, 10000]

plt.figure(figsize=(12, 8))

for i, k in enumerate(k_values, 1):
    X_k_highest = np.zeros_like(X, dtype=complex)
    highest_freq_idxs = np.argsort(X_abs)[-k:]
    for freq_idx in highest_freq_idxs:
            X_k_highest[freq_idx] = X[freq_idx]
            if freq_idx != 0:
                X_k_highest[N - freq_idx] = np.conj(X[freq_idx])
    x_k_highest = np.fft.ifft(X_k_highest)
    plt.subplot(len(k_values), 1, i)
    plt.plot(np.arange(N), x, label='Original', alpha=0.5)
    plt.plot(np.arange(N), x_k_highest, label=f'Top {k}')
    plt.legend(loc='upper right')
    plt.title(f'Aprox using top {k} frequencies')

plt.tight_layout()
plt.savefig(fname='Lab5/aprox_test.pdf')
plt.show()