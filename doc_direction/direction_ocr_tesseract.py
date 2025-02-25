import pytesseract
from PIL import Image, ImageOps
import numpy as np
import os

def process_image_orientation(img: Image.Image) -> Image.Image:
    """处理图像方向并返回校正后的图像
    
    Args:
        img: PIL.Image对象
        
    Returns:
        PIL.Image: 校正后的图像对象
    """
    try:
        # 转换为RGB模式并获取numpy数组
        img = img.convert('RGB')
        img_array = np.array(img)
        
        # 获取OSD信息
        osd_info = pytesseract.image_to_osd(
            img_array,
            config="--psm 0"
        )
        print(osd_info)
        # 解析方向信息
        rotate = int(osd_info.split("\nRotate: ")[1].split("\n")[0])
        confidence = float(osd_info.split("\nOrientation confidence: ")[1].split("\n")[0])
        
        # 根据rotate进行校正
        if confidence > 0.1:  # 只在置信度大于0.1时进行校正
            if rotate != 0:
                corrected_img = img.rotate(-rotate, expand=True)
                return corrected_img
        
        return img
            
    except Exception as e:
        print(f"方向检测失败: {str(e)}")
        raise e

def detect_and_correct_orientation(image_path: str) -> Image.Image:
    """使用Tesseract检测并校正图像方向
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        PIL.Image: 校正后的图像对象
    """
    try:
        # 打开图像并去除EXIF方向信息
        img = Image.open(image_path)
        # 使用exif_transpose移除方向标记并保留像素数据
        img = ImageOps.exif_transpose(img)
        
        # 获取OSD信息并处理
        corrected_img = process_image_orientation(img)
        
        return corrected_img
            
    except Exception as e:
        print(f"方向检测失败: {str(e)}")
        raise e


if __name__ == "__main__":
    print("图像方向校正工具（输入 q 退出）")
    # pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
    print("Tesseract 版本:", pytesseract.get_tesseract_version())
    
    while True:
        image_path = input("\n请输入图片路径：").strip().strip('\"\'')
        
        if image_path.lower() == 'q':
            print("程序已退出")
            break
            
        if not os.path.exists(image_path):
            print(f"错误：文件不存在 - {image_path}")
            continue
        try:
            corrected_img = detect_and_correct_orientation(image_path)
            corrected_img.show()
        except Exception as e:
            print(f"错误：{str(e)}")
            continue
