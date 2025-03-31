import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 创建结果目录
os.makedirs('../results', exist_ok=True)

# Harris角点检测实现
def harris_corner_detector(img_path, output_path, threshold=0.01, k=0.04, window_size=3, gaussian_sigma=1.0):
    # 读取图像并转为灰度
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # 计算梯度
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算各分量
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    
    # 高斯滤波
    Ix2 = cv2.GaussianBlur(Ix2, (window_size, window_size), gaussian_sigma)
    Iy2 = cv2.GaussianBlur(Iy2, (window_size, window_size), gaussian_sigma)
    Ixy = cv2.GaussianBlur(Ixy, (window_size, window_size), gaussian_sigma)
    
    # 计算角点响应函数
    det = Ix2 * Iy2 - Ixy ** 2
    trace = Ix2 + Iy2
    R = det - k * (trace ** 2)
    
    # 非极大值抑制
    R_max = cv2.dilate(R, None)
    mask = (R == R_max)
    R = R * mask
    
    # 阈值处理
    R[R < threshold * R.max()] = 0
    
    # 获取角点坐标
    corners = np.argwhere(R > 0)
    
    # 在原图上绘制角点
    img_out = img.copy()
    for y, x in corners:
        cv2.circle(img_out, (x, y), 3, (0, 0, 255), -1)
    
    cv2.imwrite(output_path, img_out)
    return corners

# HOG描述子计算
def compute_hog_descriptor(gray_img, keypoints, cell_size=8, block_size=2, nbins=9):
    descriptors = []
    for kp in keypoints:
        x, y = int(kp[1]), int(kp[0])  # Harris返回的是(y,x)
        half_size = cell_size * block_size // 2
        
        # 边界处理
        if x < half_size or x >= gray_img.shape[1] - half_size or y < half_size or y >= gray_img.shape[0] - half_size:
            descriptors.append(np.zeros((block_size**2 * nbins), dtype=np.float32))
            continue
        
        # 提取局部区域
        patch = gray_img[y-half_size:y+half_size, x-half_size:x+half_size]
        
        # 计算梯度
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        descriptor = []
        for i in range(0, patch.shape[0], cell_size):
            for j in range(0, patch.shape[1], cell_size):
                cell_ang = ang[i:i+cell_size, j:j+cell_size]
                cell_mag = mag[i:i+cell_size, j:j+cell_size]
                hist = np.zeros(nbins)
                for a, m in zip(cell_ang.flatten(), cell_mag.flatten()):
                    bin_idx = int(a // (180 / nbins)) % nbins
                    hist[bin_idx] += m
                hist /= np.linalg.norm(hist) + 1e-5  # 归一化
                descriptor.extend(hist)
        descriptors.append(np.array(descriptor,dtype=np.float32))
    return np.array(descriptors)

# 图像拼接函数
import cv2
import numpy as np

def stitch_images(img1, img2, M):
    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 计算img1的四个角点变换后的坐标
    corners_img1 = np.array([
        [0, 0],
        [w1-1, 0],
        [0, h1-1],
        [w1-1, h1-1]
    ], dtype=np.float32)
    
    transformed_corners = cv2.transform(corners_img1.reshape(1, -1, 2), M).reshape(-1, 2)
    
    # 合并img2的四个角点坐标（原图坐标）
    all_points = np.concatenate([
        transformed_corners,
        np.array([[0, 0], [w2-1, 0], [0, h2-1], [w2-1, h2-1]])
    ])
    
    # 计算画布尺寸
    min_x = np.floor(np.min(all_points[:, 0])).astype(int)
    max_x = np.ceil(np.max(all_points[:, 0])).astype(int)
    min_y = np.floor(np.min(all_points[:, 1])).astype(int)
    max_y = np.ceil(np.max(all_points[:, 1])).astype(int)
    
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    
    # 构造平移变换矩阵
    translation_matrix = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y]
    ], dtype=np.float32)
    
    M_combined = M
    M_combined[0,2] += translation_matrix[0,2]
    M_combined[1,2] += translation_matrix[1,2]
    # 变换两张图像到画布坐标系
    warped_img1 = cv2.warpAffine(img1, M_combined, (canvas_width, canvas_height))
    warped_img2 = cv2.warpAffine(img2, translation_matrix, (canvas_width, canvas_height))
    
    # 合并图像（简单覆盖）
    mask = (warped_img1 > 0).any(axis=2)
    warped_img2[mask] = warped_img1[mask]
    
    return warped_img2

# 处理sudoku.png
harris_corner_detector('../images/sudoku.png', '../results/sudoku_keypoints.png', threshold=0.01)

# 处理uttower图像
uttower1_corners = harris_corner_detector('../images/uttower1.jpg', '../results/uttower1_keypoints.jpg', threshold=0.01)
uttower2_corners = harris_corner_detector('../images/uttower2.jpg', '../results/uttower2_keypoints.jpg', threshold=0.01)

# 读取图像用于特征匹配
img1 = cv2.imread('../images/uttower1.jpg')
img2 = cv2.imread('../images/uttower2.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 转换关键点为KeyPoint格式
kp1 = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 20) for pt in uttower1_corners]
kp2 = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 20) for pt in uttower2_corners]

# 使用SIFT描述子
sift = cv2.SIFT_create()
kp1_sift, des1 = sift.compute(gray1, kp1)
kp2_sift, des2 = sift.compute(gray2, kp2)

# 使用HOG描述子
des1_hog = compute_hog_descriptor(gray1, uttower1_corners)
des2_hog = compute_hog_descriptor(gray2, uttower2_corners)

# 匹配特征
def match_and_draw(des1, des2, kp1, kp2, img1, img2, output_path):
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]  # 取前100个最佳匹配
    
    # 绘制匹配结果
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    cv2.imwrite(output_path, matched_img)
    
    # RANSAC计算变换矩阵
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    inliers = mask.ravel().tolist()
    
    # 绘制内点匹配
    good_matches = [m for m, i in zip(matches, inliers) if i]
    good_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
    cv2.imwrite(output_path.replace('.png', '_good.png'), good_img)
    
    # 拼接图像
    if len(good_matches) >= 3:  # 仿射变换至少需要3个点
        src_pts_good = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts_good = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        # 使用good_matches计算仿射变换矩阵
        M_good, mask_good = cv2.estimateAffine2D(src_pts_good, dst_pts_good,
                                                 method=cv2.RANSAC, ransacReprojThreshold=5.0)
        # 如果变换矩阵计算成功，则用它来拼接图像
        if M_good is not None:
            stitched = stitch_images(img1, img2, M_good)
            cv2.imwrite(output_path.replace('_match_', '_stitching_'), stitched)
            return M_good
    return None

# print(des1.shape, des2.shape, des1_hog.shape, des2_hog.shape)
# SIFT匹配和拼接
M_good_sift = match_and_draw(des1, des2, kp1_sift, kp2_sift, img1, img2, '../results/uttower_match_sift.png')

# HOG匹配和拼接
M_good_hog = match_and_draw(des1_hog, des2_hog, kp1, kp2, img1, img2, '../results/uttower_match_hog.png')

# 多图拼接（使用SIFT）
def multi_stitch(image_paths, output_path):
    base_img = cv2.imread(image_paths[0])
    for path in image_paths[1:]:
        img = cv2.imread(path)
        gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        gray_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测SIFT特征
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_base, None)
        kp2, des2 = sift.detectAndCompute(gray_new, None)
        # des1 = np.array(des1, dtype=np.float32)
        # des2 = np.array(des2, dtype=np.float32)

        # print(des1.shape, des2.shape)
        # 匹配特征
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
            M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            
            if M is not None:
                base_img = stitch_images(base_img, img, M)
    
    cv2.imwrite(output_path, base_img)

# 处理yosemite图像
yosemite_paths = ['../images/yosemite1.jpg', '../images/yosemite2.jpg', '../images/yosemite3.jpg', '../images/yosemite4.jpg']
multi_stitch(yosemite_paths, '../results/yosemite_stitching.jpg')