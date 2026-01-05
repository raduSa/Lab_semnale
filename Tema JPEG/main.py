import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets, ndimage
from scipy.fft import dctn, idctn
from skimage import data
import cv2 as cv

def get_mse(orig, jpeg):
    return np.mean((orig - jpeg) ** 2)

def rgb_to_ycbcr(img):
    img = img.astype(np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B

    return np.stack([Y, Cb, Cr], axis=2)

def ycbcr_to_rgb(img):
    img = img.astype(np.float64)
    Y  = img[:, :, 0]
    Cb = img[:, :, 1]
    Cr = img[:, :, 2]

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    rgb = np.stack([R, G, B], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

Q_chroma = [[17, 18, 24, 47, 99, 99, 99, 99,],
            [18, 21, 26, 66, 99, 99, 99, 99,],
            [24, 26, 56, 99, 99, 99, 99, 99,],
            [47, 66, 99, 99, 99, 99, 99, 99,],
            [99, 99, 99, 99, 99, 99, 99, 99,],
            [99, 99, 99, 99, 99, 99, 99, 99,],
            [99, 99, 99, 99, 99, 99, 99, 99,],
            [99, 99, 99, 99, 99, 99, 99, 99,]]

def quantize_block(block, scale):
    Q_jpeg_scaled = np.array(Q_jpeg) * scale

    # Encoding
    x = block.astype(np.float64) - 128
    y = dctn(x, norm='ortho')
    y_jpeg = Q_jpeg_scaled * np.round(y / Q_jpeg_scaled)

    # y_nnz = np.count_nonzero(y)
    # y_jpeg_nnz = np.count_nonzero(y_jpeg)
    # print('Componente in frecventa:' + str(y_nnz) + 
    #   '\nComponente in frecventa dupa cuantizare: ' + str(y_jpeg_nnz))

    # Decoding
    x_jpeg = idctn(y_jpeg, norm='ortho') + 128

    return x_jpeg

def pad_image(img):
    padded_img = img.copy()

    if img.shape[0] % 8 != 0:    
        last_line = img[-1, :].copy()
        required_lines_cnt = 8 - (img.shape[0] % 8)
        padding_lines = np.vstack([last_line] * required_lines_cnt)    
        padded_img = np.vstack((padded_img, padding_lines))
    if img.shape[1] % 8 != 0:
        last_col = padded_img[:, -1].copy()
        required_cols_cnt = 8 - (img.shape[1] % 8)
        padding_cols = np.hstack([last_col[:, None]] * required_cols_cnt)
        padded_img = np.hstack((padded_img, padding_cols))
    
    return padded_img

def get_jpeg_image(X, quality_scale_factor):
    # Get YCbCr representation
    X = rgb_to_ycbcr(X)
        
    # Encode and decode entrie image
    X = pad_image(X)
    X_jpeg = np.zeros(X.shape)

    for i in range(0, X.shape[0], 8):
        for j in range(0, X.shape[1], 8):
            block = X[i : i + 8, j : j + 8, :]        

            X_jpeg[i : i + 8, j : j + 8, 0] = quantize_block(block[:, :, 0], quality_scale_factor)
            X_jpeg[i : i + 8, j : j + 8, 1] = quantize_block(block[:, :, 1], quality_scale_factor)
            X_jpeg[i : i + 8, j : j + 8, 2] = quantize_block(block[:, :, 2], quality_scale_factor)

    return ycbcr_to_rgb(X_jpeg)

# Get RGB image
X = data.astronaut()

# Target MSE
target_MSE = 40

# Binary search for quality factor so that we get close to target MSE
q_low, q_high = 1, 5

for _ in range(20):
    q = (q_low + q_high) / 2
    X_jpeg = get_jpeg_image(X, quality_scale_factor=q)
    MSE = get_mse(X, X_jpeg)    

    if MSE < target_MSE:
        q_low = q
    else:
        q_high = q

print(f'MSE: {get_mse(X, X_jpeg)}')

plt.subplot(121).imshow(X)
plt.title('Original')
plt.subplot(122).imshow(X_jpeg)
plt.title('JPEG')
plt.show()

# Now try on video
cap = cv.VideoCapture("Tema JPEG/walter.mp4")
fps = cap.get(cv.CAP_PROP_FPS)
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(fps, width, height)
fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter(
    "Tema JPEG/output.mp4",
    fourcc,
    fps,
    (width, height)
)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Apply jpeg encoding
    frame_jpeg = get_jpeg_image(frame_rgb, quality_scale_factor=4)    
    frame_jpeg = np.clip(frame_jpeg, 0, 255).astype(np.uint8)

    # Back to BGR
    frame_bgr = cv.cvtColor(frame_jpeg, cv.COLOR_RGB2BGR)

    out.write(frame_bgr)
    print(f'Done: frame {frame_idx}')
    frame_idx += 1

cap.release()
out.release()
