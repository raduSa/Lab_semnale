import numpy as np
import matplotlib.pyplot as plt
import csv

x = np.genfromtxt('Lab5/archive/Train.csv', delimiter=',')
x = x[1:]
# print(x[:100])
# a) -> e sampled din ora in ora
# b) -> 25-08-2012 00:00 - 25-09-2014 23:00
# c) fs = 1/3600 => max freq fara aliasing = 1/7200
# d)
N = x.shape[0]
x = x[:, 2]
X = np.fft.fft(x)
X = abs(X/N)
X = X[:N//2]

Fs = 1 / 3600

f = (Fs / N) * np.linspace(0,N//2,N//2)
plt.stem(f, X)
plt.title('X')
plt.savefig(fname='Lab5/subpct_d).pdf')
plt.show()

# e) -> X[0]
cont_comp = X[0]
print(cont_comp, np.mean(x))
print(f'Componenta continua: {cont_comp}')

X = np.fft.fft(x)
X[0] = 0
x_without_DC = np.fft.ifft(X)
fig, axs = plt.subplots(2)
axs[0].plot(np.arange(N), x)
axs[0].title.set_text('Original')
axs[1].plot(np.arange(N), x_without_DC)
axs[1].title.set_text('No DC')
plt.savefig(fname='Lab5/subpct_e).pdf')
plt.tight_layout()
plt.show()

# f)
x -= cont_comp
X = np.fft.fft(x)
X = abs(X/N)
X = X[:N//2]
largest_4_freq = np.argsort(X)[-30:]
print(f'Largest frequencies:')
for i, freq_index in enumerate(reversed(largest_4_freq)):
    print(f'{i + 1}: idx-{freq_index} {f[freq_index]} Hz - {round(f[freq_index] * 3600 * 24, 2)} per day - {round(f[freq_index] * 3600 * 24 * 30, 2)} per month')

# g)
starting_sample = 2352
x += cont_comp
sample_count = 24 * 30
plt.plot(np.arange(sample_count), x[starting_sample:starting_sample+sample_count])
plt.title('1 Month: 01-12-2012 -- 01-01-2013')
plt.savefig(fname='Lab5/subpct_g).pdf')
plt.show()

# h) Normal, daca stiu frecventa de esantionare si timpul ultimului esantion atunci stiu si timpul primului esantion.
# Altfel, pot sa ma uit la esantioanele din transformata Fourier unde se regasesc spike-uri si sa incerc sa ghicesc
# carei frecvente corespund spike-urile respective (o data pe saptamna, luna etc.) si sa deduc frecventa de esantionare 
# de acolo. Acuratetea ar depinde de cat de bine pot sa intuiesc trend-urile repetitive.

# i)
filter_indexes = [108, 109, 110, 111] # indecsi asociati frecventei saptamanale

x_original = x
x -= cont_comp
X = np.fft.fft(x)

for idx in filter_indexes:
    X[idx] = 0
    X[N - idx] = 0
x_filtered = np.fft.ifft(X)

fig, axs = plt.subplots(2)
axs[0].plot(np.arange(N), x)
axs[0].title.set_text('Original')
axs[1].plot(np.arange(N), x_filtered)
axs[1].title.set_text('Weekly freq filtered out')
plt.tight_layout()
plt.show()



