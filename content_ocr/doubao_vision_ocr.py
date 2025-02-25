import requests
from typing import Dict
import json
import base64
import os
from urllib.parse import urlparse
from PIL import Image

from io import BytesIO

DETAIL_FLAG = True
DETAIL = "high" if DETAIL_FLAG else "low"
MAX_TOKENS = 12288
TEMPERATURE = 0.1
MODEL = "doubao-1.5-vision-pro-32k-250115"
PROMPT = "识别图片中的作文的题目和正文内容，注意图片可能是多张图片拼接的，需要识别出完整的作文题目和正文内容"

def compress_image(image: Image.Image, detail_flag: bool = False) -> Image.Image:
    """
    压缩图片到指定大小限制
    
    Args:
        image (Image.Image): 输入的PIL图像对象
        detail_flag (bool): 是否使用高精度模式，默认False
            True: 限制像素数小于401万
            False: 限制像素数小于104万
    
    Returns:
        Image.Image: 压缩后的图像对象
    """
    width, height = image.size
    pixel_count = width * height
    
    # 设置像素上限
    max_pixels = 4010000 if detail_flag else 1040000
    
    # 如果当前像素数小于限制，直接返回原图
    if pixel_count <= max_pixels:
        return image
    
    # 计算需要的缩放比例
    scale = (max_pixels / pixel_count) ** 0.5
    
    # 计算新的宽高
    new_width = int(width * scale)
    new_height = int(height * scale)
    print(f"压缩图片: {new_width}x{new_height}, 像素数: {new_width * new_height}, 缩放比例: {scale}")
    # 使用LANCZOS重采样方法调整图片大小
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

class DoubaoVisionAPI:
    """豆包 Vision API 客户端类"""
    
    def __init__(self, api_key: str, model: str = MODEL, prompt: str = PROMPT, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE, detail: str = DETAIL, stream: bool = False):
        """
        初始化豆包 Vision API 客户端
        
        Args:
            api_key (str): API 密钥
            model (str): 模型名称，默认使用 doubao-1.5-vision-pro-32k-250115
            prompt (str): 提示文本，默认使用PROMPT
            max_tokens (int): 最大令牌数，默认12288
            temperature (float): 温度，默认0.1
            detail (str): 细节，默认"high"
            stream (bool): 是否流式输出，默认False
        """ 
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.detail = detail
        self.stream = stream
        self.api_key = api_key
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        将本地图片转换为base64编码，在转换前会调整图片方向和大小
        
        Args:
            image_path (str): 本地图片路径
            
        Returns:
            str: base64编码后的图片数据
        """
        # 读取图片
        image = Image.open(image_path)
        
        # 压缩图片
        compressed_image = compress_image(image, True if self.detail == "high" else False)
        
        # 将压缩后的图片转换为bytes
        
        buffer = BytesIO()
        compressed_image.save(buffer, format=image.format or 'JPEG')
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_data
    
    def analyze_image(self, image_input: str) -> Dict:
        """
        分析图片内容
        
        Args:
            prompt (str): 提示文本
            image_input (str): 图片URL或本地路径
            
        Returns:
            Dict: API 响应结果
        """
        # 改进URL检测方法，正确处理Windows路径
        parsed_url = urlparse(image_input)
        # 只有当scheme是http或https等网络协议，才认为是URL
        # Windows路径如d:会被urlparse错误地识别为scheme
        is_url = parsed_url.scheme in ['http', 'https']
        
        print(f"输入类型: {'URL' if is_url else '本地文件'}")
        
        if not is_url:
            # 打印本地文件的信息
            print(f"本地文件路径: {image_input}")
            print(f"文件是否存在: {os.path.exists(image_input)}")
            if os.path.exists(image_input):
                print(f"文件大小: {os.path.getsize(image_input)} 字节")
        
        # 创建图片数据
        if is_url:
            image_data = {
                "type": "image_url",
                "image_url": {
                    "url": image_input,
                    "detail": self.detail
                }
            }
            print(f"使用URL: {image_input}")
        else:
            # 对于本地文件，使用base64编码
            base64_data = self._encode_image_to_base64(image_input)
            url = f"data:image/jpeg;base64,{base64_data}"
            image_data = {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": self.detail
                }
            }
            print(f"Base64编码前缀: {url[:50]}...")
            print(f"Base64编码长度: {len(base64_data)}")
        
        print(f"使用模型: {self.model}")
        print(f"详细程度: {self.detail}")
        print(f"最大tokens: {self.max_tokens}")
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt
                        },
                        image_data
                    ]
                }
            ]
        }
        
        print("准备发送请求...")
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            
            print(f"响应状态码: {response.status_code}")
            
            # 尝试获取响应的详细内容
            error_detail = ""
            try:
                error_json = response.json()
                error_detail = f"\n详细错误: {json.dumps(error_json, ensure_ascii=False, indent=2)}"
            except:
                error_detail = f"\n响应内容: {response.text}"
                
            # 检查状态码并抛出异常
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 请求失败: {str(e)}{error_detail}")

def main():
    api_key = "c2a758e8-f351-4f35-b292-2053481e569c"
    vision_api = DoubaoVisionAPI(api_key)
    
    print("欢迎使用豆包视觉API！")
    print("请输入图片地址（可以是网络URL或本地图片路径）")
    print("按Ctrl+C退出程序")
    
    try:
        while True:
            image_input = input("\n请输入图片地址: ").strip()
            
            # 去掉首尾的单引号或双引号
            image_input = image_input.strip("'\"")
            
            if not image_input:
                print("图片地址不能为空，请重新输入")
                continue
                
            if not urlparse(image_input).scheme and not os.path.exists(image_input):
                print("输入的本地图片不存在，请检查路径是否正确")
                continue
                
            try:
                result = vision_api.analyze_image(
                    image_input=image_input
                )
                print("\n分析结果:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"处理图片时出错: {str(e)}")
                
    except KeyboardInterrupt:
        print("\n\n感谢使用！再见！")

if __name__ == "__main__":
    main()
