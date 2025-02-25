import base64
import dashscope

API_KEY = "sk-6a985295fd59458e8ae8689a8fe58381"
PROMPT = "识别图片中的作文的题目和正文内容，注意图片可能是多张图片拼接的或者是试卷书籍分开的两部分，需要识别出完整的作文题目和正文内容"

def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

while True:
    file_path = input("请输入图片路径（直接回车退出）: ").strip().strip('\'"')  # 去除首尾空格和引号
    if not file_path:
        break
    
    # 构建消息内容
    content = [
        {"image": f"data:image/jpeg;base64,{image_to_base64(file_path)}"},
        {"text": PROMPT}
    ]
    
    # 调用API，官网 dashscope 调用方式 api 写的parameters 中传入 一个什么参数，可以使用更高清的图片。
    # 当前其实是图片 1176x1176 像素的效果。他应该会先压缩图片或 token
    # 和 doubao 的 vision 模型类似，没有尝试，是当前效果已经还可以了。
    response = dashscope.MultiModalConversation.call(
        api_key=API_KEY,
        model='qwen-vl-max-latest',
        messages=[{"role": "user", "content": content}]
    )
    
    
    
    if response.status_code == 200:
        # print("\n=== 解析结果 ===")
        # print("完整输出结构:")
        # print(response.output)
        
        print("\n文本内容:")
        for content in response.output.choices[0].message.content:
            if 'text' in content:
                print(content['text'])
    else:
        print("\n=== 错误详情 ===")
        print(f"错误码: {response.code}")
        print(f"错误信息: {response.message}")
        print(f"请求ID: {response.request_id}")
        if response.usage:
            print(f"消耗Token: {response.usage.get('total_tokens', '无')}")
