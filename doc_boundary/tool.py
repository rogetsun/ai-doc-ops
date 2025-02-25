#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys

class DocumentDetector:
    """文档边界检测器，用于检测图像中的文档边界"""
    
    def __init__(self, blur_ksize=5, canny_threshold1=75, canny_threshold2=200):
        """
        初始化文档边界检测器
        
        参数:
            blur_ksize: 高斯模糊核大小
            canny_threshold1: Canny边缘检测的低阈值
            canny_threshold2: Canny边缘检测的高阈值
        """
        self.blur_ksize = blur_ksize
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
    
    def detect(self, image):
        """
        检测图像中的文档边界
        
        参数:
            image: 输入图像，BGR格式
            
        返回:
            contour: 检测到的文档边界，按面积排序取最大的一个，形状为(4,2)的numpy数组
            如果未检测到有效边界，返回None
        """
        # 增加对手写文档的特殊处理
        # 对于手写文档，增加阈值对比度
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        
        # 膨胀边缘
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 没有找到轮廓
        if not contours:
            return None
        
        # 按面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 取最大的轮廓
        largest_contour = contours[0]
        
        # 轮廓近似，得到多边形
        epsilon = 0.05 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 如果多边形有四个点，认为是文档
        if len(approx) == 4:
            # 确保点按左上、右上、右下、左下的顺序排列
            pts = approx.reshape(4, 2)
            rect = self._order_points(pts)
            return rect
        
        # 如果近似多边形不是四边形，尝试找到最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.intp)
        box = self._order_points(box)
        return box
    
    def _order_points(self, pts):
        """
        将四个点按照左上、右上、右下、左下的顺序排列
        
        参数:
            pts: 四个点的坐标，形状为(4,2)的numpy数组
            
        返回:
            有序的四个点，形状为(4,2)的numpy数组
        """
        # 创建一个包含四个坐标点的数组
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 按坐标和计算左上和右下
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        
        # 按坐标差计算右上和左下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        
        return rect

    def _remove_background_texture(self, gray):
        """移除格子背景纹理"""
        # 使用形态学操作移除背景纹理
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # 自适应阈值处理
        binary = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        return binary


class DocumentTransformer:
    """文档透视变换器，用于矫正倾斜的文档"""
    
    def __init__(self):
        """初始化文档透视变换器"""
        pass
    
    def transform(self, image, contour, target_width=None, target_height=None):
        """
        对文档进行透视变换，矫正到矩形
        
        参数:
            image: 输入图像
            contour: 文档的边界轮廓，形状为(4,2)的numpy数组
            target_width: 目标宽度，如果为None，则根据轮廓自动计算
            target_height: 目标高度，如果为None，则根据轮廓自动计算
            
        返回:
            变换后的图像
        """
        if contour is None or len(contour) != 4:
            return image
        
        # 计算目标尺寸
        if target_width is None or target_height is None:
            width_a = np.sqrt(((contour[1][0] - contour[0][0]) ** 2) + ((contour[1][1] - contour[0][1]) ** 2))
            width_b = np.sqrt(((contour[2][0] - contour[3][0]) ** 2) + ((contour[2][1] - contour[3][1]) ** 2))
            max_width = max(int(width_a), int(width_b))
            
            height_a = np.sqrt(((contour[3][0] - contour[0][0]) ** 2) + ((contour[3][1] - contour[0][1]) ** 2))
            height_b = np.sqrt(((contour[2][0] - contour[1][0]) ** 2) + ((contour[2][1] - contour[1][1]) ** 2))
            max_height = max(int(height_a), int(height_b))
            
            target_width = target_width or max_width
            target_height = target_height or max_height
        
        # 设置目标轮廓点
        dst = np.array([
            [0, 0],                      # 左上
            [target_width - 1, 0],       # 右上
            [target_width - 1, target_height - 1],  # 右下
            [0, target_height - 1]       # 左下
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(contour.astype(np.float32), dst)
        
        # 应用透视变换
        warped = cv2.warpPerspective(image, M, (target_width, target_height))
        
        return warped
    
    def draw_contour(self, image, contour, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制轮廓
        
        参数:
            image: 输入图像
            contour: 文档边界，形状为(4,2)的numpy数组
            color: 轮廓颜色
            thickness: 线宽
            
        返回:
            绘制了轮廓的图像
        """
        if contour is None or len(contour) != 4:
            return image
        
        result = image.copy()
        
        # 绘制四个点和连线
        for i in range(4):
            pt1 = tuple(contour[i].astype(int))
            pt2 = tuple(contour[(i + 1) % 4].astype(int))
            cv2.line(result, pt1, pt2, color, thickness)
            cv2.circle(result, pt1, 5, color, -1)
        
        return result


class EdgeEnhancer:
    """文档边缘增强器，用于增强文档边缘的清晰度"""
    
    def __init__(self, kernel_size=3, sigma=1.0, amount=1.0):
        """
        初始化边缘增强器
        
        参数:
            kernel_size: 锐化核大小
            sigma: 锐化高斯模糊的sigma值
            amount: 锐化强度
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount
    
    def enhance(self, image):
        """
        增强图像边缘
        
        参数:
            image: 输入图像
            
        返回:
            边缘增强后的图像
        """
        # 使用USM锐化（Unsharp Masking）
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        sharpened = cv2.addWeighted(image, 1.0 + self.amount, blurred, -self.amount, 0)
        return sharpened
    
    def enhance_adaptive(self, image):
        """
        自适应边缘增强，只增强边缘区域
        
        参数:
            image: 输入图像
            
        返回:
            自适应边缘增强后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 检测边缘
        edges = cv2.Canny(gray, 50, 150)
        
        # 膨胀边缘区域
        edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        
        # 创建边缘掩码
        edge_mask = edges_dilated / 255.0
        
        # 应用USM锐化
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        sharpened = cv2.addWeighted(image, 1.0 + self.amount, blurred, -self.amount, 0)
        
        # 只增强边缘区域
        if len(image.shape) == 3:
            # 转换掩码为3通道
            edge_mask_3ch = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2)
            # 使用掩码混合原图和锐化图
            result = (sharpened * edge_mask_3ch + image * (1 - edge_mask_3ch)).astype(np.uint8)
        else:
            # 使用掩码混合原图和锐化图
            result = (sharpened * edge_mask + image * (1 - edge_mask)).astype(np.uint8)
        
        return result


class RegionCropper:
    """文档区域裁剪器，用于裁剪文档的指定区域"""
    
    def __init__(self):
        """初始化区域裁剪器"""
        pass
    
    def crop_by_ratio(self, image, top_ratio=0.0, right_ratio=0.0, bottom_ratio=0.0, left_ratio=0.0):
        """
        按比例裁剪图像边缘
        
        参数:
            image: 输入图像
            top_ratio: 顶部裁剪比例 (0.0-1.0)
            right_ratio: 右侧裁剪比例 (0.0-1.0)
            bottom_ratio: 底部裁剪比例 (0.0-1.0)
            left_ratio: 左侧裁剪比例 (0.0-1.0)
            
        返回:
            裁剪后的图像
        """
        height, width = image.shape[:2]
        
        top = int(height * top_ratio)
        bottom = int(height * (1 - bottom_ratio))
        left = int(width * left_ratio)
        right = int(width * (1 - right_ratio))
        
        return image[top:bottom, left:right]
    
    def crop_by_pixels(self, image, top=0, right=0, bottom=0, left=0):
        """
        按像素裁剪图像边缘
        
        参数:
            image: 输入图像
            top: 顶部裁剪像素数
            right: 右侧裁剪像素数
            bottom: 底部裁剪像素数
            left: 左侧裁剪像素数
            
        返回:
            裁剪后的图像
        """
        height, width = image.shape[:2]
        
        return image[top:height-bottom, left:width-right]
    
    def crop_to_content(self, image, padding=0):
        """
        裁剪到内容区域，去除多余的空白
        
        参数:
            image: 输入图像
            padding: 内容周围的填充像素数
            
        返回:
            裁剪后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 查找非零像素的边界
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # 添加填充
        height, width = image.shape[:2]
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        # 裁剪
        return image[y:y+h, x:x+w]


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    保持纵横比调整图像大小
    
    参数:
        image: 输入图像
        width: 目标宽度，如果为None，则根据高度按比例计算
        height: 目标高度，如果为None，则根据宽度按比例计算
        inter: 插值方法
        
    返回:
        调整大小后的图像
    """
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    
    # 如果宽度和高度都为None，则返回原始图像
    if width is None and height is None:
        return image
    
    # 如果宽度为None，则根据高度按比例计算宽度
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    # 否则，根据宽度按比例计算高度
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    # 调整图像大小
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized


def rotate_image(image, angle, center=None, scale=1.0):
    """
    旋转图像
    
    参数:
        image: 输入图像
        angle: 旋转角度（度），正值表示逆时针旋转
        center: 旋转中心，如果为None，则使用图像中心
        scale: 缩放因子
        
    返回:
        旋转后的图像
    """
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    
    # 如果旋转中心未指定，则默认为图像中心
    if center is None:
        center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 应用旋转
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated


def auto_orientation(image):
    """
    自动检测并矫正文档方向
    
    参数:
        image: 输入图像
        
    返回:
        矫正方向后的图像和旋转角度
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 霍夫线变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # 如果没有检测到线段，返回原图和0度角
    if lines is None:
        return image, 0
    
    # 计算线段的角度
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 避免除以零
        if x2 == x1:
            continue
        angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
        angles.append(angle)
    
    # 如果没有有效角度，返回原图和0度角
    if not angles:
        return image, 0
    
    # 计算角度的直方图，找出最常见的角度
    hist, bins = np.histogram(angles, bins=36, range=(-90, 90))
    center_bin = (bins[:-1] + bins[1:]) / 2
    dominant_angle = center_bin[np.argmax(hist)]
    
    # 如果主导角度接近0度或90度，则不旋转
    if abs(dominant_angle) < 5 or abs(abs(dominant_angle) - 90) < 5:
        return image, 0
    
    # 计算需要旋转的角度
    if dominant_angle < 45 and dominant_angle > -45:
        rotation_angle = -dominant_angle
    elif dominant_angle >= 45:
        rotation_angle = 90 - dominant_angle
    else:  # dominant_angle <= -45
        rotation_angle = -90 - dominant_angle
    
    # 旋转图像
    rotated = rotate_image(image, rotation_angle)
    
    return rotated, rotation_angle


def cleanup_path(path):
    """清理输入路径，移除首尾的引号"""
    path = path.strip()
    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
        path = path[1:-1]
    return path


def remove_checkered_background(image):
    """
    移除格子状背景，突出文档边界
    
    参数:
        image: 输入图像
        
    返回:
        处理后的图像，突出了文档边界
    """
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯滤波平滑图像
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用自适应阈值分割前景和背景
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    
    # 使用形态学操作移除小的噪点
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 应用闭操作连接相邻区域
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    
    # 找到最大连通区域（假设是文档）
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    # 创建掩码
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], 0, 255, -1)
    
    # 扩大掩码区域，确保包含文档边缘
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    
    # 应用掩码到原图
    result = image.copy()
    result[mask == 0] = [255, 255, 255]  # 将背景设为白色
    
    return result


def process_image(image_path):
    """处理图像的完整流程"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：文件 '{image_path}' 不存在")
        return False
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 '{image_path}'")
        return False
    
    # 获取输出文件名
    filename, ext = os.path.splitext(image_path)
    
    # 保存原始图像副本
    cv2.imwrite(f"{filename}_original{ext}", image)
    print(f"✓ 已保存原始图像: {filename}_original{ext}")
    
    # 增加预处理步骤：增强对比度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # 转回彩色图像以便后续处理
    enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"{filename}_contrast{ext}", enhanced_image)
    print(f"✓ 已保存对比度增强的图像: {filename}_contrast{ext}")
    
    # 继续使用增强后的图像进行后续处理
    image = enhanced_image
    
    # 调整图像大小
    if max(image.shape[0], image.shape[1]) > 1500:
        image = resize_image(image, width=1500)
    
    # 自动纠正方向
    oriented_image, angle = auto_orientation(image)
    if abs(angle) > 0:
        cv2.imwrite(f"{filename}_oriented{ext}", oriented_image)
        print(f"✓ 已保存方向校正后的图像: {filename}_oriented{ext}")
    
    # 修改检测器参数，应对格子背景
    detector = DocumentDetector(blur_ksize=5, canny_threshold1=30, canny_threshold2=150)
    
    # 增加额外的图像处理步骤，去除格子背景
    # 转换为灰度图
    gray_oriented = cv2.cvtColor(oriented_image, cv2.COLOR_BGR2GRAY)
    
    # 进行背景抑制，增强文档边缘（可选步骤）
    bg_suppressed = cv2.morphologyEx(gray_oriented, cv2.MORPH_TOPHAT, 
                                    np.ones((15, 15), np.uint8))
    cv2.imwrite(f"{filename}_bg_suppressed{ext}", bg_suppressed)
    print(f"✓ 已保存背景抑制后的图像: {filename}_bg_suppressed{ext}")
    
    # 检测文档边界
    contour = detector.detect(oriented_image)
    
    if contour is None:
        print("未检测到文档边界")
        return False
    
    # 绘制边界
    transformer = DocumentTransformer()
    marked_image = transformer.draw_contour(oriented_image, contour)
    cv2.imwrite(f"{filename}_boundary{ext}", marked_image)
    print(f"✓ 已保存边界标记的图像: {filename}_boundary{ext}")
    
    # 透视变换
    warped = transformer.transform(oriented_image, contour)
    cv2.imwrite(f"{filename}_transformed{ext}", warped)
    print(f"✓ 已保存透视变换后的图像: {filename}_transformed{ext}")
    
    # 边缘增强
    enhancer = EdgeEnhancer()
    enhanced = enhancer.enhance_adaptive(warped)
    cv2.imwrite(f"{filename}_enhanced{ext}", enhanced)
    print(f"✓ 已保存边缘增强后的图像: {filename}_enhanced{ext}")
    
    # 去除格子背景
    bg_removed = remove_checkered_background(enhanced)
    cv2.imwrite(f"{filename}_bg_removed{ext}", bg_removed)
    print(f"✓ 已保存去除格子背景的图像: {filename}_bg_removed{ext}")
    
    # 使用处理后的图像
    image = bg_removed
    
    # 裁剪内容区域
    cropper = RegionCropper()
    result = cropper.crop_to_content(image, padding=10)
    cv2.imwrite(f"{filename}_final{ext}", result)
    print(f"✓ 已保存最终处理后的图像: {filename}_final{ext}")
    
    return True


def main():
    print("==== 文档区域边缘操作工具 ====")
    print("输入 'q' 或 'exit' 退出程序\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入图片路径: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("程序已退出")
                break
            
            # 清理路径
            image_path = cleanup_path(user_input)
            
            # 处理图片
            success = process_image(image_path)
            if success:
                print("\n处理完成！")
        except KeyboardInterrupt:
            print("\n程序已被用户中断")
            break
        except Exception as e:
            print(f"\n处理过程中出错: {str(e)}")


if __name__ == "__main__":
    main() 