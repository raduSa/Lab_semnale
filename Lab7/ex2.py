from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

X = datasets.face(gray=True)
X_orig = X.copy()
print(X.shape)

limit_SNR = 1.01

curr_SNR = float('inf')
k = 50
iteration = 0

while curr_SNR > limit_SNR:

    Y = np.fft.fft2(X)
    Y_flat = Y.flatten()
    
    top_k_freq = np.argsort(np.abs(Y_flat))[-k:]
    Y_flat[top_k_freq] = 0    
    Y = Y_flat.reshape(X.shape[0], X.shape[1])
    X = np.fft.ifft2(Y).real

    curr_SNR = np.linalg.norm(X_orig) / np.linalg.norm(X_orig - X)
    print(f'Current SNR: {curr_SNR}')
    iteration +=1

plt.imshow(X, cmap=plt.cm.gray)
plt.title(f'Compresed after removing top {iteration * k}')
plt.savefig(fname=f'Lab7/ex2_fig1.pdf')
plt.show()