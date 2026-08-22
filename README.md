# Canpoint OCR

## future

1. 增加用户账号系统及收费模式
2. 增加后台log
3. 增加输入数据校验

## 修改

### 修改 8.06

1. 增加“照片”文本行弯曲图像修复
2. 增加Photo/PDF文本整合为题目功能，提供相应接口
2. 修复mathpix结果格式bug

### 重构 7.29

1. 去除弃用代码及模块
2. 算法模块整合，去除不同进程分别调用算法模块的运作方式
3. 缩小try范围，只包含在mathpix调用以及服务请求中
4. 重做数据持久化模块，方便采集mathpix结果
5. 添加config全局配置，包括模型路径，mathpix开关，服务url端口等
6. weight路径整合，提供下载脚本，方便部署
7. 重做web接口

## 如何使用

### 启动服务
python server.py

### 发送请求
import requests  
url = 'http://<服务器ip>:<服务端口>/photo_ocr_test'  
files = {'image': open('<图片路径>', 'rb')}  
response = requests.post(url, files=files)  

