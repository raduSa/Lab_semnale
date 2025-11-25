from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

X = datasets.face(gray=True)
pixel_noise = 100

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise

SNR_before_denoise = np.linalg.norm(X) / np.linalg.norm(noise)
print(f'SNR before denoise {SNR_before_denoise}')

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy')

Y_noisy = np.fft.fftshift(np.fft.fft2(X_noisy))

plt.subplot(2, 2, 3)
plt.imshow(20*np.log(np.abs(Y_noisy)), cmap=plt.cm.gray)
plt.title('Frequency')

Y_filtered = Y_noisy.copy()
mask = np.zeros_like(Y_filtered, dtype='uint8')
rows, cols = mask.shape
center_x, center_y = cols // 2, rows // 2
r = 25
cv.circle(mask, (center_x, center_y), r, 1, -1)

# print(mask)

plt.subplot(2, 2, 4)
plt.imshow(mask, cmap=plt.cm.gray)
plt.title('Mask')

Y_filtered = Y_filtered * mask

X_denoised = np.fft.ifft2(np.fft.fftshift(Y_filtered)).real

SNR_after_denoise = np.linalg.norm(X) / np.linalg.norm(X - X_denoised)
print(f'SNR after denoise {SNR_after_denoise}')

plt.subplot(2, 2, 2)
plt.imshow(X_denoised, cmap=plt.cm.gray)
plt.title('Denoised')

plt.tight_layout()  
plt.savefig(fname=f'Lab7/ex3_fig1.pdf')
plt.show()