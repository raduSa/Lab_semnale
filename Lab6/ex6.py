import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

x = np.genfromtxt('Lab5/archive/Train.csv', delimiter=',')

# a)
# 3 days: 1776,07-11-2012 00:00 - 1847,09-11-2012 23:00
x = x[1776:1848, 2]

plt.plot(np.arange(72), x)
plt.title('Original signal (3 days)')
plt.savefig(fname=f'Lab6/ex6_fig1.pdf')
plt.show()

# b)
plt.figure(figsize=(16, 12))
for i, w in enumerate([5, 9, 13, 17]):
    x_filtered = np.convolve(x, np.ones(w), 'valid') / w
    plt.subplot(4, 1, i+1)
    plt.title(f'W = {w}')
    plt.plot(np.arange(len(x_filtered)), x_filtered)

plt.tight_layout()
plt.savefig(fname=f'Lab6/ex6_fig2.pdf')
plt.show()

# c) Fs = 1 / 3600 Hz => frecventa fundamentala e 1 / 3600 / 72 -> 1 data la 3 zile
#                        cea mai inalta frecventa pe care o pot analiza = frecv Nyquist = Fs / 2 -> 1 data la 2h
#                        o sa aleg cutoff freq o data pe zi (pt ca asa vreau) -> 1/12 frecv Nyquist = Fs / 24

# d)
f_cutoff = 1 / 3600 / 24
f_Nyquist = 1 / 3600 / 2
Wn = f_cutoff / f_Nyquist
b_butter, a_butter = scipy.signal.butter(5, Wn, btype='low')
b_cheb, a_cheb = scipy.signal.cheby1(5, 5, Wn, btype='low')

# e)
# as alege Butterworth pt ca nu imi distorsioneaza frecventele din banda de trecere (linia e mai putin plata)
fig, axs = plt.subplots(2)
axs[0].plot(np.arange(72), scipy.signal.filtfilt(b_butter, a_butter, x), label='Butterworth')
axs[0].plot(np.arange(72), x, label='Original')
axs[0].legend(loc='upper right')

axs[1].plot(np.arange(72), scipy.signal.filtfilt(b_cheb, a_cheb, x), label='Chebyshev')
axs[1].plot(np.arange(72), x, label='Original')
axs[1].legend(loc='upper right')

plt.savefig(fname=f'Lab6/ex6_fig3.pdf')
plt.show()

# f) 
plt.figure(figsize=(16, 12))

params = [(order, rp) for order in [3, 5, 7] for rp in[2, 5, 10]]

for i, (order, rp) in enumerate(params):
    b_butter, a_butter = scipy.signal.butter(order, Wn, btype='low')
    b_cheb, a_cheb = scipy.signal.cheby1(order, rp, Wn, btype='low')
    
    plt.subplot(3, 3, i+1)
    plt.plot(np.arange(72), scipy.signal.filtfilt(b_butter, a_butter, x), label='Butterworth', color='green')
    plt.plot(np.arange(72), x, label='Original', color='blue')
    plt.plot(np.arange(72), scipy.signal.filtfilt(b_cheb, a_cheb, x), label='Chebyshev', color='red')
    plt.title(f'Order={order} rp={rp}')
    plt.legend(loc='upper right')

plt.savefig(fname=f'Lab6/ex6_fig4.pdf')
plt.show()