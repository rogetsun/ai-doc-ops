import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket  # 使用websocket_client
from PIL import Image
import io
import threading

# 讯飞星火视觉识别理解模型

appid = "985dcf71"    #填写控制台中获取的 APPID 信息
api_secret = "MjFlOGU1NDNiNWFjNjM5YzRhOTNkMTYx"   #填写控制台中获取的 APISecret 信息
api_key ="d6fbf713d52c171f1fc3c2157aeea100"    #填写控制台中获取的 APIKey 信息
imageunderstanding_url = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"#云端环境的服务地址
PROMPT = "识别图片中的作文的题目和正文内容，注意图片可能是多张图片拼接的或者是试卷书籍分开的两部分，需要识别出完整的作文题目和正文内容"



class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, imageunderstanding_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(imageunderstanding_url).netloc
        self.path = urlparse(imageunderstanding_url).path
        self.ImageUnderstanding_url = imageunderstanding_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.ImageUnderstanding_url + '?' + urlencode(v)
        #print(url)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws,one,two):
    print(" ")


# 收到websocket连接建立的处理
def on_open(ws):
    data = json.dumps(gen_params(appid=ws.appid, messages=ws.messages))
    print("\n发送给模型的消息:")
    
    # 处理打印内容，隐藏base64图片数据
    print_data = json.loads(data)
    if "payload" in print_data and "message" in print_data["payload"] and "text" in print_data["payload"]["message"]:
        messages = print_data["payload"]["message"]["text"]
        for msg in messages:
            if msg.get("content_type") == "image":
                msg["content"] = "[图片数据]"
    
    print(json.dumps(print_data, indent=2, ensure_ascii=False))
    ws.send(data)


# 收到websocket消息的处理
def on_message(ws, message):
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        global answer
        answer += content
        if status == 2:
            ws.close()
            # 设置完成事件
            ws.done_event.set()


def process_image(img_path):
    try:
        # 读取并编码图片
        imagedata = open(img_path, 'rb').read()
        print(f"图片大小：{len(imagedata)/1024/1024:.2f} MB")
        
        # 构建消息列表
        messages = [
            {"role": "user", "content": str(base64.b64encode(imagedata), 'utf-8'), "content_type":"image"},
            {"role": "user", "content": PROMPT}
        ]
        
        # 初始化返回结果
        global answer
        answer = ""
        
        # 创建事件对象
        done_event = threading.Event()
        
        # 调用API获取结果
        main(appid, api_key, api_secret, imageunderstanding_url, messages, done_event)
        
        # 等待完成
        done_event.wait()
        return answer
        
    except FileNotFoundError:
        print(f"错误：文件 '{img_path}' 不存在")
        return None
    except Exception as e:
        print(f"处理文件出错: {str(e)}")
        return None

def gen_params(appid, messages):
    """
    通过appid和消息列表来生成请参数
    """
    data = {
        "header": {
            "app_id": appid
        },
        "parameter": {
            "chat": {
                "domain": "imagev3",
                "temperature": 0.5,
                "top_k": 1,
                "max_tokens": 8192,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": messages
            }
        }
    }
    return data

def main(appid, api_key, api_secret, imageunderstanding_url, messages, done_event):
    wsParam = Ws_Param(appid, api_key, api_secret, imageunderstanding_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.messages = messages  # 改名
    ws.done_event = done_event
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})


if __name__ == '__main__':
    while True:
        print("\n请输入图片路径(输入q退出): ")
        img_path = input().strip()
        
        if img_path.lower() == 'q':
            break
            
        # 去除首尾的引号
        if (img_path.startswith('"') and img_path.endswith('"')) or \
           (img_path.startswith("'") and img_path.endswith("'")):
            img_path = img_path[1:-1]
        
        # 处理图片并获取结果
        result = process_image(img_path)
        if result:
            print("\n识别结果:")
            print(result)
