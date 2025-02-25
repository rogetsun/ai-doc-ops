import cv2
import numpy as np
from paddleocr import PaddleOCR
import math

class BoundaryDetectorPaddleOCR:
    def __init__(self, use_gpu=False, lang="ch"):
        """
        初始化PaddleOCR文档边界检测器
        
        参数:
            use_gpu: 是否使用GPU加速
            lang: 语言，默认中文
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
    
    def detect_boundary(self, img_path):
        """
        检测图像中的文档边界
        
        参数:
            img_path: 图像路径
            
        返回:
            corners: 文档的四个角点坐标 [左上, 右上, 右下, 左下]
            status: 检测状态，True表示成功
        """
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                return None, False
            
            # 使用PaddleOCR进行文本检测
            result = self.ocr.ocr(img_path, cls=True)
            
            if not result or len(result) == 0 or not result[0]:
                return None, False
            
            # 获取所有文本框的坐标
            boxes = []
            for line in result[0]:
                # PaddleOCR返回的是文本框的四个角点坐标
                points = line[0]
                boxes.append(points)
            
            if not boxes:
                return None, False
            
            # 计算文档边界
            return self._find_document_boundary(boxes, img.shape), True
            
        except Exception as e:
            print(f"检测文档边界时出错: {e}")
            return None, False
    
    def _find_document_boundary(self, boxes, img_shape):
        """
        根据文本框计算文档边界
        
        参数:
            boxes: 文本框坐标列表
            img_shape: 图像尺寸
            
        返回:
            corners: 文档的四个角点坐标 [左上, 右上, 右下, 左下]
        """
        # 将所有文本框的点合并成一个大列表
        all_points = []
        for box in boxes:
            for point in box:
                all_points.append(point)
        
        all_points = np.array(all_points)
        
        # 使用凸包算法找到最外围的点
        hull = cv2.convexHull(all_points)
        hull = hull.reshape(-1, 2)
        
        # 找到最接近矩形的四个点
        # 可以使用多边形逼近算法
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # 如果逼近结果不是4个点，则需要特殊处理
        if len(approx) != 4:
            # 如果点太多，找到边界矩形
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            approx = np.int0(box)
        
        # 对四个点进行排序，使其符合[左上, 右上, 右下, 左下]的顺序
        corners = self._order_points(approx.reshape(-1, 2))
        
        # 确保边界在图像范围内
        h, w = img_shape[:2]
        corners = np.clip(corners, [0, 0], [w-1, h-1])
        
        return corners.astype(int)
    
    def _order_points(self, pts):
        """
        对四个点进行排序，使其符合[左上, 右上, 右下, 左下]的顺序
        
        参数:
            pts: 四个点的坐标
            
        返回:
            rect: 排序后的坐标
        """
        # 初始化
        rect = np.zeros((4, 2), dtype="float32")
        
        # 计算每个点的坐标和
        s = pts.sum(axis=1)
        # 左上角的点，坐标和最小
        rect[0] = pts[np.argmin(s)]
        # 右下角的点，坐标和最大
        rect[2] = pts[np.argmax(s)]
        
        # 计算每个点的坐标差
        diff = np.diff(pts, axis=1)
        # 右上角的点，y-x差值最小
        rect[1] = pts[np.argmin(diff)]
        # 左下角的点，y-x差值最大
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def perspective_transform(self, img_path, corners, output_size=(800, 1100)):
        """
        对图像进行透视变换，提取文档区域
        
        参数:
            img_path: 图像路径
            corners: 文档的四个角点坐标 [左上, 右上, 右下, 左下]
            output_size: 输出图像的尺寸 (宽, 高)
            
        返回:
            warped: 变换后的图像
            status: 变换状态，True表示成功
        """
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                return None, False
            
            # 设置目标点
            dst = np.array([
                [0, 0],
                [output_size[0] - 1, 0],
                [output_size[0] - 1, output_size[1] - 1],
                [0, output_size[1] - 1]
            ], dtype="float32")
            
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
            
            # 进行透视变换
            warped = cv2.warpPerspective(img, M, output_size)
            
            return warped, True
            
        except Exception as e:
            print(f"透视变换时出错: {e}")
            return None, False
    
    def draw_boundary(self, img_path, corners):
        """
        在图像上绘制检测到的文档边界
        
        参数:
            img_path: 图像路径
            corners: 文档的四个角点坐标 [左上, 右上, 右下, 左下]
            
        返回:
            result_img: 绘制边界后的图像
            status: 绘制状态，True表示成功
        """
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                return None, False
            
            # 绘制边界
            result_img = img.copy()
            cv2.polylines(result_img, [corners], True, (0, 255, 0), 3)
            
            # 绘制角点
            for i, point in enumerate(corners):
                cv2.circle(result_img, tuple(point), 10, (0, 0, 255), -1)
                cv2.putText(result_img, str(i), tuple(point), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return result_img, True
            
        except Exception as e:
            print(f"绘制边界时出错: {e}")
            return None, False

def detect_document_boundary(img_path, use_gpu=False, lang="ch", draw=False, transform=False):
    """
    检测图像中的文档边界
    
    参数:
        img_path: 图像路径
        use_gpu: 是否使用GPU加速
        lang: 语言，默认中文
        draw: 是否绘制边界
        transform: 是否进行透视变换
        
    返回:
        result: 取决于参数，可能是角点坐标、绘制边界后的图像或透视变换后的图像
        status: 处理状态，True表示成功
    """
    detector = BoundaryDetectorPaddleOCR(use_gpu=use_gpu, lang=lang)
    
    # 检测边界
    corners, status = detector.detect_boundary(img_path)
    if not status:
        return None, False
    
    # 根据参数返回不同的结果
    if draw:
        return detector.draw_boundary(img_path, corners)
    elif transform:
        return detector.perspective_transform(img_path, corners)
    else:
        return corners, True

def cleanup_path(path):
    """清理输入路径，移除首尾的引号"""
    path = path.strip()
    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
        path = path[1:-1]
    return path


def process_image(image_path, use_gpu=False, lang="ch"):
    """使用PaddleOCR处理图像的完整流程"""
    import os
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：文件 '{image_path}' 不存在")
        return False
    
    # 获取输出文件名
    filename, ext = os.path.splitext(image_path)
    
    # 初始化检测器
    detector = BoundaryDetectorPaddleOCR(use_gpu=use_gpu, lang=lang)
    
    # 检测文档边界
    corners, status = detector.detect_boundary(image_path)
    if not status:
        print("未检测到文档边界")
        return False
    
    print(f"检测到的文档边界角点坐标：")
    print("左上:", corners[0])
    print("右上:", corners[1])
    print("右下:", corners[2])
    print("左下:", corners[3])
    
    # 绘制边界
    marked_image, status = detector.draw_boundary(image_path, corners)
    if status:
        cv2.imwrite(f"{filename}_boundary{ext}", marked_image)
        print(f"✓ 已保存边界标记的图像: {filename}_boundary{ext}")
    
    # 透视变换
    warped, status = detector.perspective_transform(image_path, corners)
    if status:
        cv2.imwrite(f"{filename}_transformed{ext}", warped)
        print(f"✓ 已保存透视变换后的图像: {filename}_transformed{ext}")
    
    return True


def main():
    print("==== PaddleOCR文档边界检测工具 ====")
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
            
            # 询问是否使用GPU
            use_gpu_input = input("是否使用GPU加速 (y/n，默认n): ").strip().lower()
            use_gpu = use_gpu_input == 'y'
            
            # 询问语言
            lang_input = input("识别语言 (ch/en，默认ch): ").strip().lower()
            lang = lang_input if lang_input in ['ch', 'en'] else 'ch'
            
            print("开始处理图像...")
            
            # 处理图片
            success = process_image(image_path, use_gpu=use_gpu, lang=lang)
            if success:
                print("\n处理完成！")
        except KeyboardInterrupt:
            print("\n程序已被用户中断")
            break
        except Exception as e:
            print(f"\n处理过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 