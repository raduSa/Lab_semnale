import numpy as np
import matplotlib.pyplot as plt

n, d = 20, 4
x = np.random.rand(n) * 90 + 10
y = np.concatenate((x[-4:], x[:n - d]))


print(f'Correlation: {np.fft.ifft(np.fft.fft(x).conj() * np.fft.fft(y)).real}')
print(f'd= {np.argmax(np.fft.ifft(np.fft.fft(x).conj() * np.fft.fft(y).real))}')

# Teoretic, ar trebui sa dea la fel mereu, practic pot sa apara probleme daca imparti la nr apropiat de 0
print(f'Division: {np.fft.ifft(np.fft.fft(y) / np.fft.fft(x)).real}')
print(f'd= {np.argmax(np.fft.ifft(np.fft.fft(y) / np.fft.fft(x)).real)}')