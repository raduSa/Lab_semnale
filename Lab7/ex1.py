from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = datasets.face(gray=True)
# print(X.shape)
# plt.imshow(X, cmap=plt.cm.gray)
# plt.show()

Y = np.fft.fft2(X)
# print(Y.shape)
freq_db = 20*np.log10(abs(Y))

# plt.imshow(freq_db)
# plt.colorbar()
# plt.show()

rotate_angle = 45
X45 = ndimage.rotate(X, rotate_angle)
# plt.imshow(X45, cmap=plt.cm.gray)
# plt.show()

Y45 = np.fft.fft2(X45)
# plt.imshow(20*np.log10(abs(Y45)))
# plt.colorbar()
# plt.show()

freq_x = np.fft.fftfreq(X.shape[1])
freq_y = np.fft.fftfreq(X.shape[0])

# plt.stem(freq_x, freq_db[:][0])
# plt.show()

freq_cutoff = 120

Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2
# plt.imshow(X_cutoff, cmap=plt.cm.gray)
# plt.show()

pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
# plt.imshow(X, cmap=plt.cm.gray)
# plt.title('Original')
# plt.show()
# plt.imshow(X_noisy, cmap=plt.cm.gray)
# plt.title('Noisy')
# plt.show()

# ex_1
# signal_1
func1 = lambda x, y: np.sin(2*np.pi*x + 3*np.pi*y)

X = [[0] * 100 for _ in range(100)]
N = 100
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
for i, line in enumerate(x):
    for j, col in enumerate(y):
        X[i][j] = func1(line, col)
X = np.array(X)

plt.subplot(1, 2, 1)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Img1')

# In frecventa ar trebui sa apara 2 puncte
Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))
plt.subplot(1, 2, 2)
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr1')
plt.savefig(fname=f'Lab7/ex1_fig1.pdf')
plt.show()

threshold = np.max(np.abs(Y))
coords = np.argwhere(np.abs(Y) == threshold)

print(coords)

# signal_2
func2 = lambda x, y: np.sin(4*np.pi*x) + np.cos(6*np.pi*y)

for i, line in enumerate(x):
    for j, col in enumerate(y):
        X[i][j] = func2(line, col)
X = np.array(X)

plt.subplot(1, 2, 1)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Img2')

# In frecventa imi apar 2 puncte pe linia 0, conjugate, pt sinusoida de pe axa Ox
# pe coloana 0, 2 puncte conjugate petnru sinusoida de pe axa Oy
Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))
plt.subplot(1, 2, 2)
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr2')
plt.savefig(fname=f'Lab7/ex1_fig2.pdf')
plt.show()

# signal_3
Y = [[0] * 100 for _ in range(100)]
Y[0][5] = Y[0][100 - 5] = 1
Y = np.array(Y)

freq_db = 20*np.log10(abs(Y))
plt.subplot(1, 2, 1)
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr3')

X = np.fft.ifft2(Y)
# Imi apare doar o sinusoida pe axa Ox de frecventa 5
plt.subplot(1, 2, 2)
plt.imshow(np.real(X), cmap=plt.cm.gray)
plt.title('Img3')
plt.savefig(fname=f'Lab7/ex1_fig3.pdf')
plt.show()

# signal_4
Y = [[0] * 100 for _ in range(100)]
Y[5][0] = Y[100 - 5][0] = 1
Y = np.array(Y)

freq_db = 20*np.log10(abs(Y))
plt.subplot(1, 2, 1)
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr4')

X = np.fft.ifft2(Y)
# Imi apare acum pe Oy, tot de frecventa 5
plt.subplot(1, 2, 2)
plt.imshow(np.real(X), cmap=plt.cm.gray)
plt.title('Img4')
plt.savefig(fname=f'Lab7/ex1_fig4.pdf')
plt.show()

#signal_5
Y = [[0] * 100 for _ in range(100)]
Y[5][5] = Y[100 - 5][100 - 5] = 1
Y = np.array(Y)

freq_db = 20*np.log10(abs(Y))
plt.subplot(1, 2, 1)
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr5')

X = np.fft.ifft2(Y)
# apar linii diagonale (de fapt cos(5*2pi*x + 5*2pi*y))
plt.subplot(1, 2, 2)
plt.imshow(np.real(X), cmap=plt.cm.gray)
plt.title('Img5')
plt.savefig(fname=f'Lab7/ex1_fig5.pdf')
plt.show()