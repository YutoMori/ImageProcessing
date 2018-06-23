# -*- coding: utf-8 -*-
# author : Yuto Mori

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像をグレースケールで読み込み
img_gray = cv2.imread("lenna.png", 0)

# 高速フーリエ変換(2次元)
fft_img = np.fft.fft2(img_gray)

# 第一象限と第三象限、第二象限と第四象限を入れ替える
fft_img = np.fft.fftshift(fft_img)

# パワースペクトル計算
mag = 20 * np.log(np.abs(fft_img))

# フーリエ変換した画像のサイズ
h, w = img_gray.shape

length = img_gray.shape[0]
center = length/2
filter_low = np.zeros(img_gray.shape)

# ローパスフィルタの設定
R = 50
for i in range(0, length):
    for j in range(0, length):
        if(i-center)*(i-center) + (j-center)*(j-center) < R*R:
            filter_low[i][j] = 1

# ローパスフィルタを適用する
lowpass_img = fft_img * filter_low

# 逆高速フーリエ変換(2次元)
ifft_img = np.fft.ifft2(lowpass_img)

# 表示するために複素数を絶対値に変換する
ifft_img = np.abs(ifft_img)

# パワースペクトル計算
mag_low = 20 * np.log(np.abs(lowpass_img))

### ヒストグラム作成
hist_lenna = [0] * 256
hist_lowlenna = [0] * 256

# 輝度値に対応する配列に1を足していく
for i in range(0, len(img_gray)):
    for j in range(0, len(img_gray[0])):
        hist_lenna[img_gray[i, j]] += 1

int_ifft_img = np.array(ifft_img, dtype='int64')
for i in range(0, len(int_ifft_img)):
    for j in range(0, len(int_ifft_img[0])):
        hist_lowlenna[int_ifft_img[i, j]] += 1


### 画像描写について
plt.figure()
plt.subplot(121),plt.imshow(img_gray, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(mag, cmap='gray')
plt.title('FFT image'), plt.xticks([]), plt.yticks([])

plt.figure()
plt.subplot(121),plt.imshow(mag_low, cmap='gray')
plt.title('FFT Image through Low Pass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(ifft_img, cmap='gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])

plt.figure()
plt.title("Histgram of Lenna")
plt.plot(hist_lenna)
plt.xlabel("Pixel value")
plt.ylabel("Number of pixels")
plt.xlim(0, 255)

plt.figure()
plt.title("Histgram of Lenna throgh low pass")
plt.plot(hist_lowlenna)
plt.xlabel("Pixel value")
plt.ylabel("Number of pixels")
plt.xlim(0, 255)

plt.show()
