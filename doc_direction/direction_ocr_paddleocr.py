from paddleocr import PaddleOCR
import os
import argparse

# 使用百度 PaddleOCR 识别文档方向。效果不好，弃用

def detect_image_orientation(image_path):
    """
    检测图片方向（增强版）
    :param image_path: 图片文件路径
    :return: 方向角度（0, 90, 180, 270）或错误信息
    """
    try:
        if not os.path.exists(image_path):
            return f"文件不存在: {image_path}"
            
        # 初始化OCR实例并启用方向检测（显式指定方向分类参数）
        ocr = PaddleOCR(use_angle_cls=True,
                        cls_model_dir='/Users/uv/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer', 
                        label_list=['0', '90', '180', '270'],
                        lang='ch')
        
        # 进行方向检测（优化参数配置）
        result = ocr.ocr(image_path, cls=True, det=False, rec=False)
        
        # 解析方向检测结果（增强健壮性）
        if result and isinstance(result, list) and len(result) > 0:
            try:
                # 获取第一个检测结果的方向角度
                # 解析方向分类结果（格式示例：[[[('cls', 90), 0.99]]]）
                print("调试信息 - 原始结果结构:", result)  # 添加调试输出
                if result and isinstance(result[0], list) and len(result[0]) > 0:
                    first_result = result[0][0]  # 获取第一个检测结果
                    if isinstance(first_result, (list, tuple)) and len(first_result) >= 2:
                        # 直接解析 first_result 结构 [标签, 置信度]
                        angle_label = str(first_result[0])  # 获取类别标签
                        confidence = float(first_result[1])  # 获取置信度
                        
                        # 根据实际模型输出调整映射关系（支持所有四个方向）
                        angle_mapping = {'0': 0, '90': 90, '180': 180, '270': 270}
                        
                        # 添加详细的调试信息
                        print(f"调试信息 - 解析结果: 标签={angle_label}, 置信度={confidence:.2f}")
                        
                        # 检查置信度阈值
                        if confidence < 0.6:
                            return f"低置信度 ({confidence:.2f})，建议人工复核"
                            
                        # 返回映射结果或错误提示
                        if angle_label in angle_mapping:
                            return angle_mapping[angle_label]
                        else:
                            return f"检测到未支持的角度标签: {angle_label}\n支持的角度: 0°, 90°, 180°, 270°"
                return "无效的检测结果格式"
            except (IndexError, TypeError, ValueError) as e:
                return f"结果解析失败: {str(e)}"
            
        return "未检测到有效方向信息"
        
    except Exception as e:
        return f"方向检测失败: {str(e)}"

if __name__ == "__main__":
    print("图片方向检测工具（输入 q 退出）")
    while True:
        image_path = input("\n请输入图片路径：").strip().strip('\"\'')  # 去除首尾空格和引号
        
        if image_path.lower() == 'q':
            print("程序已退出")
            break
            
        if not os.path.exists(image_path):
            print(f"错误：文件不存在 - {image_path}")
            continue
            
        result = detect_image_orientation(image_path)
        
        if isinstance(result, int):
            print("\n检测结果：")
            print(f"旋转角度: {result}°")
            print("调整建议：")
            rotations = {
                0: "无需旋转",
                90: "向左旋转90度 (逆时针)",
                180: "旋转180度",
                270: "向右旋转90度 (顺时针)"
            }
            print(f"→ {rotations.get(result, '未知旋转角度')}")
        else:
            print(f"错误：{result}")
