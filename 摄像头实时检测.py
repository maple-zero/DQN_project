import cv2
import numpy as np
import time
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO

import matplotlib.pyplot as plt


import torch
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

#导入模型
model = YOLO('C:\\Users\h2973\Desktop\hand\Train_Custom_Dataset-main\Train_Custom_Dataset-main\keypoints\\4-train\handpro\s_pretrain2\weights\\best.pt')

# 切换计算设备
model.to(device)
# model.cpu()  # CPU
# model.cuda() # GPU

# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 2                   # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size':1,         # 字体大小
    'font_thickness':2,    # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-10,        # Y 方向，文字偏移距离，向下为正
}

# 关键点 BGR 配色
hand_kpt_color_map = {
    0:{'name':'中冲', 'color':[0, 0, 255], 'radius':6},      
    1:{'name':'商阳', 'color':[0, 0, 255], 'radius':6},      
    2:{'name':'中魁', 'color':[0, 0, 255], 'radius':6},      
    3:{'name':'十宣', 'color':[0, 0, 255], 'radius':6},
    4:{'name':'关冲', 'color':[0, 0, 255], 'radius':6},
    5:{'name':'少冲', 'color':[0, 0, 255], 'radius':6},
    6:{'name':'少泽', 'color':[0, 0, 255], 'radius':6},
    7:{'name':'小骨空', 'color':[0, 0, 255], 'radius':6},
    8:{'name':'液门', 'color':[0, 0, 255], 'radius':6},
    9:{'name':'合谷', 'color':[0, 0, 255], 'radius':6},
}

palm_kpt_color_map = {
    0:{'name':'中冲', 'color':[0, 0, 255], 'radius':6},      
    1:{'name':'少商', 'color':[0, 0, 255], 'radius':6},      
    2:{'name':'劳宫', 'color':[0, 0, 255], 'radius':6},      
    3:{'name':'少府', 'color':[0, 0, 255], 'radius':6},
    4:{'name':'鱼际', 'color':[0, 0, 255], 'radius':6},
    5:{'name':'太渊', 'color':[0, 0, 255], 'radius':6},
    6:{'name':'大陵', 'color':[0, 0, 255], 'radius':6},
    7:{'name':'神门', 'color':[0, 0, 255], 'radius':6},
    8:{'name':'大肠', 'color':[0, 0, 255], 'radius':6},
    9:{'name':'肾', 'color':[0, 0, 255], 'radius':6},
}
# 点类别文字
kpt_labelstr = {
    'font_size':1.5,             # 字体大小
    'font_thickness':3,       # 字体粗细
    'offset_x':10,             # X 方向，文字偏移距离，向右为正
    'offset_y':0,            # Y 方向，文字偏移距离，向下为正
}

#置信度
confidence = 0.92

#存储手背节点
hand_arr = np.empty((10,3))

#存储手心节点
palm_arr = np.empty((10,3))

boxxy = np.empty(4)
#绘制中文
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    """
    img:opecv格式
    cv2显示中文字符
    """
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


#逐帧处理函数
def process_frame(img_bgr,sum):
    
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''
    global hand_arr,palm_arr,boxxy
    # 记录该帧开始处理的时间
    start_time = time.time()
    
    results = model(img_bgr, verbose=False) # verbose设置为False，不单独打印每一帧预测结果
    
    #预测框的置信度
    bboxes_conf=results[0].boxes.conf
    num_bbox = len(results[0].boxes.cls)
    
    # 预测框的 xyxy 坐标
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32') 
    
    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy().astype('uint32')
        
    for idx in range(num_bbox): # 遍历每个框
        
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx] 
        # print(bbox_xyxy)
        # 获取框的预测类别
        bbox_cls = results[0].boxes.cls
        bbox_label = results[0].names[0]
        bbox_label2 = results[0].names[1] 
        # 画框
        if bboxes_conf[idx] > confidence:
            #优化框抖动
          if sum == 0:
            boxxy = bbox_xyxy
          if sum != 0:
            boxxy = (boxxy+bbox_xyxy)/2
            boxxy = np.round(boxxy).astype(int)
          img_bgr = cv2.rectangle(img_bgr, (boxxy[0], boxxy[1]), (boxxy[2], boxxy[3]), bbox_color, bbox_thickness)
          if bbox_cls[0].item() == 0.0:
            # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])
             
          if bbox_cls[0].item() == 1.0:
            # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            img_bgr = cv2.putText(img_bgr, bbox_label2, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

        bbox_keypoints = bboxes_keypoints[idx] # 该框所有关键点坐标和置信度
        
        if bbox_cls[0].item() == 0.0:
             # 画手背的关键点
            if sum == 0:
                    hand_arr = bbox_keypoints
                   
            if sum != 0:
                    hand_arr = np.mean([hand_arr,bbox_keypoints],axis=0)
                    hand_arr = np.round(hand_arr).astype(int)
                    
            for kpt_id in hand_kpt_color_map:

            # 获取该关键点的颜色、半径、XY坐标
                kpt_color = hand_kpt_color_map[kpt_id]['color']
                kpt_radius = hand_kpt_color_map[kpt_id]['radius']
                kpt_x = hand_arr[kpt_id][0]
                kpt_y = hand_arr[kpt_id][1]
                
            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
                if bboxes_conf[idx] > confidence:
                  img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
                  
                  # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
                  #kpt_label = str(kpt_id) # 写关键点类别 ID（二选一）
                  kpt_label = str(hand_kpt_color_map[kpt_id]['name']) # 写关键点类别名称（二选一）
                  #img_bgr = cv2.putText(img_bgr, kpt_label, (kpt_x+kpt_labelstr['offset_x'], kpt_y+kpt_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color, kpt_labelstr['font_thickness'])
                  img_bgr = cv2AddChineseText(img_bgr, kpt_label, (kpt_x+kpt_labelstr['offset_x'], kpt_y+kpt_labelstr['offset_y']), (100, 255, 100), 15)
        
        if bbox_cls[0].item() == 1.0:
             # 画手心的关键点
            if sum == 0:
                    palm_arr = bbox_keypoints
            if sum != 0:
                    palm_arr = np.mean([palm_arr,bbox_keypoints],axis=0)
                    palm_arr = np.round(palm_arr).astype(int)
            for kpt_id in palm_kpt_color_map:

            # 获取该关键点的颜色、半径、XY坐标
                kpt_color = palm_kpt_color_map[kpt_id]['color']
                kpt_radius = palm_kpt_color_map[kpt_id]['radius']
                kpt_x = palm_arr[kpt_id][0]
                kpt_y = palm_arr[kpt_id][1]
                
            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
                if bboxes_conf[idx] > confidence:
                  img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

                  # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
                  #kpt_label = str(kpt_id) # 写关键点类别 ID（二选一）
                  kpt_label = str(palm_kpt_color_map[kpt_id]['name']) # 写关键点类别名称（二选一）
                  #img_bgr = cv2.putText(img_bgr, kpt_label, (kpt_x+kpt_labelstr['offset_x'], kpt_y+kpt_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color, kpt_labelstr['font_thickness'])
                  img_bgr = cv2AddChineseText(img_bgr, kpt_label, (kpt_x+kpt_labelstr['offset_x'], kpt_y+kpt_labelstr['offset_y']), (100, 255, 100), 15)
    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1/(end_time - start_time)

    # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    FPS_string = 'FPS  '+str(int(FPS)) # 写在画面上的字符串
    img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)
    
    return img_bgr



# 获取摄像头，传入0表示获取系统默认摄像头,realsense RGB为2
cap = cv2.VideoCapture(0)

# 打开cap
cap.open(0)
sum = 0
# 无限循环，直到break被触发
while cap.isOpened():
    
    # 获取画面
    success, frame = cap.read()
    
    if not success: # 如果获取画面不成功，则退出
        print('获取画面不成功，退出')
        break
    
    ## 逐帧处理
    frame = process_frame(frame,sum)
    sum = sum+1
    # 展示处理后的三通道图像
    cv2.imshow('my_window',frame)
    
    key_pressed = cv2.waitKey(60) # 每隔多少毫秒毫秒，获取键盘哪个键被按下
    # print('键盘上被按下的键：', key_pressed)

    if key_pressed in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
        break
    
# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()