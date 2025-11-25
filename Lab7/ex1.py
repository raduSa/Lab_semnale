from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = datasets.face(gray=True)
print(X.shape)
# plt.imshow(X, cmap=plt.cm.gray)
# plt.show()

Y = np.fft.fft2(X)
print(Y.shape)
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

# 1
# signal_1
func1 = lambda x, y: np.sin(2*np.pi*x + 3*np.pi*y)

X = [[0] * 1000 for _ in range(1000)]
for line in range(1000):
    for col in range(1000):
        X[line][col] = func1(line, col)
X = np.array(X)
print(X.shape)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Img1')
# plt.show()

Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr1')
# plt.show()

# signal_2
func2 = lambda x, y: np.sin(4*np.pi*x) + np.cos(6*np.pi*y)

X = [[0] * 1000 for _ in range(1000)]
for line in range(1000):
    for col in range(1000):
        X[line][col] = func2(line, col)
X = np.array(X)
print(X)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Img2')
# plt.show()

Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr2')
# plt.show()

# signal_3
Y = [[0] * 1000 for _ in range(1000)]
Y[0][5] = Y[0][1000 - 5] = 1
Y = np.array(Y)

freq_db = 20*np.log10(abs(Y))
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr3')
# plt.show()

X = np.fft.ifft2(Y)

plt.imshow(np.real(X), cmap=plt.cm.gray)
plt.title('Img3')
# plt.show()

# signal_4
Y = [[0] * 1000 for _ in range(1000)]
Y[5][0] = Y[1000 - 5][0] = 1
Y = np.array(Y)

freq_db = 20*np.log10(abs(Y))
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr4')
# plt.show()

X = np.fft.ifft2(Y)

plt.imshow(np.real(X), cmap=plt.cm.gray)
plt.title('Img4')
# plt.show()

#signal_5
Y = [[0] * 1000 for _ in range(1000)]
Y[5][5] = Y[1000 - 5][1000 - 5] = 1
Y = np.array(Y)

freq_db = 20*np.log10(abs(Y))
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectr5')
# plt.show()

X = np.fft.ifft2(Y)

plt.imshow(np.real(X), cmap=plt.cm.gray)
plt.title('Img5')
# plt.show()

plt.show()