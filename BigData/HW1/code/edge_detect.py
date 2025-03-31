import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
img = cv2.imread('../results/image.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊降噪（Canny和Laplacian需要）
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Sobel边缘检测
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.addWeighted(
    cv2.convertScaleAbs(sobel_x), 0.5, 
    cv2.convertScaleAbs(sobel_y), 0.5, 0
)

# Scharr边缘检测（改进版Sobel）
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
scharr_combined = cv2.addWeighted(
    cv2.convertScaleAbs(scharr_x), 0.5,
    cv2.convertScaleAbs(scharr_y), 0.5, 0
)

# Laplacian边缘检测
laplacian = cv2.Laplacian(blur, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# Canny边缘检测
canny = cv2.Canny(blur, 100, 200)

# 使用Matplotlib展示结果
plt.figure(figsize=(15,10))

# 原图
plt.subplot(231), plt.imshow(img_rgb)
plt.title('Original Image'), plt.axis('off')

# Sobel
plt.subplot(232), plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection'), plt.axis('off')

# Scharr
plt.subplot(233), plt.imshow(scharr_combined, cmap='gray')
plt.title('Scharr Edge Detection'), plt.axis('off')

# Laplacian
plt.subplot(234), plt.imshow(laplacian_abs, cmap='gray')
plt.title('Laplacian Edge Detection'), plt.axis('off')

# Canny
plt.subplot(235), plt.imshow(canny, cmap='gray')
plt.title('Canny Edge Detection'), plt.axis('off')

plt.tight_layout()
plt.savefig('../results/edge_detection.png')
plt.show()