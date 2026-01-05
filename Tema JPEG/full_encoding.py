import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets, ndimage
from scipy.fft import dctn, idctn
from skimage import data
import cv2 as cv
from huffman import *
from writer import *

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

def quantize_block(block, scale, Q):
    Q_jpeg_scaled = np.array(Q) * scale

    # Encoding
    x = block.astype(np.float64) - 128
    y = dctn(x, norm='ortho')
    y_jpeg = np.round(y / Q_jpeg_scaled).astype(int)

    return y_jpeg

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

def get_DCT_coeffs(X, quality_scale_factor):
    # Get YCbCr representation
    X = rgb_to_ycbcr(X)
        
    # Get coeffs
    X = pad_image(X)
    DCT_coeffs = np.zeros(X.shape)

    for i in range(0, X.shape[0], 8):
        for j in range(0, X.shape[1], 8):
            block = X[i : i + 8, j : j + 8, :]        

            DCT_coeffs[i : i + 8, j : j + 8, 0] = quantize_block(block[:, :, 0], quality_scale_factor, Q_jpeg)
            DCT_coeffs[i : i + 8, j : j + 8, 1] = quantize_block(block[:, :, 1], quality_scale_factor, Q_chroma)
            DCT_coeffs[i : i + 8, j : j + 8, 2] = quantize_block(block[:, :, 2], quality_scale_factor, Q_chroma)

    return DCT_coeffs

def get_zigzag_pattern(n=8):
    indices = list()
    for diag_idx in range(2 * n - 1):
        if diag_idx % 2 == 1:
            for i in range(diag_idx + 1):
                j = diag_idx - i
                if i < n and j < n:
                    indices.append((i, j))
        else:
            for i in range(diag_idx, -1, -1):
                j = diag_idx - i
                if i < n and j < n:
                    indices.append((i, j))
    return indices

zigzag = get_zigzag_pattern()
# print(zigzag)

def apply_zigzag(block):
    return np.array([block[i, j] for i, j in zigzag])

# print(np.array([i + j for i in range(8) for j in range(8)]).reshape((8, 8)))
# print(apply_zigzag(np.array([i + j for i in range(8) for j in range(8)]).reshape((8, 8))))

def get_bit_size(x):
    # Size = bits required to represent the absolute value of x
    if x == 0: 
        return 0
    return int(np.floor(np.log2(np.abs(x))) + 1)

# print(get_bit_size(7), get_bit_size(0), get_bit_size(8))

def RLE_encode(coeffs):
    ZRL = (15, 0, 0)
    EOB = (0, 0, 0)

    res = list()
    zero_run = 0

    for c in coeffs:        
        if c == 0:
            zero_run += 1
            if zero_run == 16:
                res.append(ZRL)
                zero_run = 0
        else:
            size = get_bit_size(c)
            res.append((zero_run, size, c))
            zero_run = 0

    # Remove any trailing ZRL symbols and replace with EOB
    if zero_run > 0 or res[-1] == ZRL:
        while res and res[-1] == ZRL:
            res.pop()
        res.append(EOB)

    return res

def get_symbols_for_block(block, prev_DC):
    block_flat = apply_zigzag(block).astype(int)

    # DC symbol
    DC_coeff = block_flat[0]
    DC_diff = DC_coeff - prev_DC
    DC_symbol = (get_bit_size(DC_diff), DC_diff)

    # AC symbols
    AC_symbols = RLE_encode(block_flat[1:])

    return DC_coeff, DC_symbol, AC_symbols

def amplitude_bits(value, size):
    if value >= 0:
        return value
    return (1 << size) - 1 + value

def encode_block(bitwriter, DC_symbol, AC_symbols, DC_table, AC_table):
    # DC
    size, diff = DC_symbol
    code, length = DC_table[size]
    bitwriter.write_bits(code, length)

    if size > 0:
        bitwriter.write_bits(amplitude_bits(diff, size), size)

    # AC
    for run, size, value in AC_symbols:
        symbol = (run << 4) | size
        code, length = AC_table[symbol]        
        bitwriter.write_bits(code, length)

        if size > 0:
            bitwriter.write_bits(amplitude_bits(value, size), size)

if __name__ == '__main__':
    # Get RGB image
    X = data.astronaut()

    DCT_coeffs = get_DCT_coeffs(X, quality_scale_factor=1)

    # Encode all blocks

    prev_Y_DC = prev_Cb_DC = prev_Cr_DC = 0
    bitwriter = Writer()

    # for i in range(0, DCT_coeffs.shape[0], 8):
    #     for j in range(0, DCT_coeffs.shape[1], 8):
    # Just one block
    i = j = 0
    block = DCT_coeffs[i : i + 8, j : j + 8, :]

    Y_coeffs = block[:, :, 0]
    Cb_coeffs = block[:, :, 1]
    Cr_coeffs = block[:, :, 2]

    prev_Y_DC, Y_DC_symbol, Y_AC_symbols = get_symbols_for_block(Y_coeffs, prev_Y_DC)
    prev_Cb_DC, Cb_DC_symbol, Cb_AC_symbols = get_symbols_for_block(Cb_coeffs, prev_Cb_DC)
    prev_Cr_DC, Cr_DC_symbol, Cr_AC_symbols = get_symbols_for_block(Cr_coeffs, prev_Cr_DC)

    print(f'Block {i // 8 + j // 8}: ')
    print(f'Y: {Y_DC_symbol} {Y_AC_symbols}')
    print(f'Cb: {Cb_DC_symbol} {Cb_AC_symbols}')
    print(f'Cr: {Cr_DC_symbol} {Cr_AC_symbols}')

    encode_block(bitwriter, Y_DC_symbol, Y_AC_symbols, DC_LUMA, AC_LUMA)
    encode_block(bitwriter, Cb_DC_symbol, Cb_AC_symbols, DC_CHROMA, AC_CHROMA)
    encode_block(bitwriter, Cr_DC_symbol, Cr_AC_symbols, DC_CHROMA, AC_CHROMA)

    bitwriter.flush()
    entropy_data = bitwriter.buffer

    print(f'Entropy coded data: {entropy_data.hex()}')