import os
# os.environ["TESSDATA_PREFIX"] = r"C:\Users\sixon\AppData\Local\Programs\Tesseract-OCR\tessdata"

import pytesseract
from PIL import Image


# 如果是 Windows，可能需要手动设置 tesseract.exe 路径,或者已经加到环境变量了
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 转换为灰度
    img = cv2.bitwise_not(img)  # 反转颜色（黑字变白字）
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)  # 二值化
    return Image.fromarray(img)

def detect_text_orientation(image_path):
    print("hehe ",image_path)
    img = preprocess_image(image_path)
    osd = pytesseract.image_to_osd(img)
    print(osd)
    for line in osd.split("\n"):
        if "Orientation in degrees" in line:
            clockwise_rotation_degrees = int(line.split(":")[-1].strip())
            print(f"检测到文本方向需要顺时针旋转{clockwise_rotation_degrees}度来校正")
            return clockwise_rotation_degrees
    return None


def correct_image_orientation(image_path):
    img = Image.open(image_path)
    clockwise_angle = detect_text_orientation(image_path)

    if clockwise_angle and clockwise_angle > 0:
        img = img.rotate(clockwise_angle, expand=True)  # 逆时针旋转检测到的顺时针角度
        dir_name = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        output_path = os.path.join(dir_name, "corrected_" + base_name)
        img.save(output_path, exif=b'')
        print(f"已执行逆时针旋转 {clockwise_angle}° 进行校正，保存为 {output_path}")
        print("开始识别方向校准后的信息：")
        detect_text_orientation(output_path)
    else:
        print("图片无需旋转")




image_path = input("请输入图片路径：").strip('\"\'')
correct_image_orientation(image_path)
