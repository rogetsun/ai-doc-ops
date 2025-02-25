# AI-Doc-Ops

文档处理与OCR操作工具集，专注于智能文档分析与文本识别。

## 项目简介

AI-Doc-Ops 是一套综合性的文档处理工具集，主要用于解决文档方向检测和OCR文本识别等常见问题。该工具集整合了多种先进的AI模型，适用于各类文档处理场景。

## 功能模块

### 文档方向检测 (doc_direction)

自动检测文档的方向并进行校正，确保文档以正确的方向呈现。

- **PaddleOCR方向检测** - 基于PaddleOCR框架的方向检测算法 (`direction_ocr_paddleocr.py`)
- **Tesseract角度检测** - 使用Tesseract引擎的文档角度检测工具 (`direction_ocr_tesseract.py`)

### 内容OCR识别 (content_ocr)

将图像中的文字内容转换为可编辑的文本格式。

- **豆包视觉OCR** - 基于豆包(Doubao)视觉模型的OCR功能 (`doubao_vision_ocr.py`)，中文效果更好
- **通义千问视觉大模型OCR** - 利用通义千问VL Max大模型进行OCR识别 (`qwen_vl_max_ocr.py`)，英语效果更好

## 安装步骤

1. 克隆仓库到本地

```
git clone https://github.com/yourusername/AI-Doc-Ops.git
cd AI-Doc-Ops
```

2. 安装依赖包

```
pip install -r requirements.txt
```

## 使用示例

### 文档方向检测

```
from doc_direction.direction_ocr_paddleocr import detect_direction

# 检测文档方向
direction = detect_direction("path/to/your/document.jpg")
print(f"文档方向: {direction} 度")
```

### OCR文本识别

```
from content_ocr.doubao_vision_ocr import recognize_text

# 识别文档中的文字
result = recognize_text("path/to/your/document.jpg")
print(f"识别结果: {result}")
```

## 依赖环境

- Python 3.7+
- PaddleOCR
- Tesseract
- 其他依赖请查看 `requirements.txt`

## 贡献指南

欢迎提交问题和功能请求。如果您想贡献代码，请遵循以下步骤：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 许可证

本项目采用 [MIT](LICENSE) 许可证。

## 联系方式

如有问题，请通过 [issues](https://github.com/yourusername/AI-Doc-Ops/issues) 页面联系我们。
