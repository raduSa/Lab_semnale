import numpy as np
import matplotlib.pyplot as plt

N = 5

p = np.int64(np.random.rand(N) * 100)
q = np.int64(np.random.rand(N) * 100)

print(f'Using convolution: {np.convolve(p, q)}')

p_padded = np.pad(p, (0, N-1), 'constant', constant_values=(0,))
q_padded = np.pad(q, (0, N-1), 'constant', constant_values=(0,))

P = np.fft.fft(p_padded)
Q = np.fft.fft(q_padded)

R = P * Q

r = np.fft.ifft(R)

print(f'Using FFT: {r.real}')