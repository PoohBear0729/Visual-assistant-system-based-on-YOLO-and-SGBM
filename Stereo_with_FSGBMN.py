import cv2
import numpy as np
import time
import math

import torch

from ultralytics import YOLO
import threading
from threading import Thread
import os
import pandas as pd
import torchvision.transforms as transforms
from visual_system import FSBGM
import torchvision.models as models
import torch.nn as nn

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 转换为张量并归一化到 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
imageWidth = 1280
imageHeight = 720
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet18(pretrained=True)

modules = list(resnet.children())[:-2]  # 去掉最后的全连接层和全局平均池化层
resnet50_modified = nn.Sequential(*modules)

for param in resnet50_modified.parameters():
    param.requires_grad = False

imageSize = (imageWidth, imageHeight)

# 相机内参和畸变系数
left_camera_matrix = np.array([[914.59397, 0, 672.40331],
                               [0, 914.65162, 367.76235],
                               [0, 0, 1]], dtype=np.float64)

left_distortion = np.array([0.02463, -0.05272, 0.00043, -0.00070, 0], dtype=np.float64)

right_camera_matrix = np.array([[914.85377, 0, 651.59317],
                                [0, 922.28511, 361.48557],
                                [0, 0, 1]], dtype=np.float64)

right_distortion = np.array([0.03507, -0.06988, 0.00056, -0.00074, 0], dtype=np.float64)

# 旋转矩阵和平移向量
R_Vector = np.array([-0.00372, 0.00444, -0.00337], dtype=np.float64)
R, _ = cv2.Rodrigues(R_Vector)
print(R)
T = np.array([-217.76538, 2.25904, 3.86207], dtype=np.float64)

size = (1280, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
print(Q)


class Stereo_Thread(Thread):
    def __init__(self):
        super().__init__()
        self.left = None
        self.right = None
        self.threeD = []
        self.lock = threading.Lock()

    def set_data(self, left_image, right_image):
        with self.lock:
            self.left = left_image
            self.right = right_image

    def get_data(self, x, y):  # 输入目标在原图像的x, y坐标
        with self.lock:
            x_rectified = left_map1[y, x][0]
            y_rectified = left_map1[y, x][1]
            x_rectified = int(x_rectified)
            y_rectified = int(y_rectified)
            if 0 <= y_rectified < self.threeD.shape[0] and 0 <= x_rectified < self.threeD.shape[1]:
                distance = math.sqrt(
                    self.threeD[y_rectified][x_rectified][0] ** 2 +
                    self.threeD[y_rectified][x_rectified][1] ** 2 +
                    self.threeD[y_rectified][x_rectified][2] ** 2)
                distance = distance / 1000.0

        return distance

    def run(self):
        with self.lock:
            if self.left is None or self.right is None:
                print("左右图像未设置!")
                return
        imgL = cv2.cvtColor(self.left, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(self.right, cv2.COLOR_BGR2GRAY)

        # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
        # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
        img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

        # 转换为opencv的BGR格式

        # ------------------------------------SGBM算法----------------------------------------------------------
        #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
        #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
        #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
        #                               取16、32、48、64等
        #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
        #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
        # ------------------------------------------------------------------------------------------------------
        blockSize = 6
        stereo = cv2.StereoSGBM_create(minDisparity=1,
                                       numDisparities=256,
                                       blockSize=blockSize,
                                       P1=100,
                                       P2=1000,
                                       disp12MaxDiff=1,
                                       preFilterCap=1,
                                       uniquenessRatio=10,
                                       speckleWindowSize=100,
                                       speckleRange=100,
                                       mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        # 计算视差
        disparity = stereo.compute(img1_rectified, img2_rectified)
        # 计算三维坐标数据值
        self.threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
        # 计算出的threeD，需要乘以16，才等于现实中的距离
        self.threeD = self.threeD * 16


# 图像尺寸


# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------

# 加载视频文件

device = 'cuda' if torch.cuda.is_available() else 'cpu'
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
model = YOLO("F:\\algorithm\\ultralytics-main\\yolov8n.pt")
FSGBM_model = FSBGM(resnet50_modified)  # 使用您自己的backbone
FSGBM_model.load_state_dict(torch.load("F:\\algorithm\\ultralytics-main\\fsgbm.pth"))
FSGBM_model.to(device)  # 移动模型到GPU

FSGBM_model.eval()  # 切换到评估模式
# 读取视频
fps = 0.0
ret, frame = capture.read()
print(ret)
if not ret:
    print("error")
    exit(-1)
else:
    print(frame.shape)

# 摄像头捕捉

while True:
    # 开始计时
    stereo_thread = Stereo_Thread()
    t1 = time.time()

    # 读取帧，ret为True表示成功读取
    ret, frame = capture.read()
    if not ret:
        print("Failed to capture frame")
        break

    # 切割为左右两张图片
    frame1 = frame[0:720, 0:1280]  # 左图
    frame2 = frame[0:720, 1280:2560]  # 右
    # 启动SGBM线程进行深度测量
    stereo_thread.set_data(frame1, frame2)
    stereo_thread.start()

    # 使用模型进行检测
    results = model.track(frame1, device='0', save=False, conf=0.25)
    annotated_frame = results[0].plot()  # 带距离标注的帧

    # 获取检测框的信息，假设只取第一个目标
    boxes = results[0].boxes.xywh.cpu()
    if len(boxes) > 0:
        box = boxes[0]  # 只处理第一个检测框
        x_center, y_center, width, height = box.tolist()

        # 等待SGBM测距线程完成
        stereo_thread.join()

        # 获取SGBM测距数据
        distance = stereo_thread.get_data(int(x_center), int(y_center))
        # 检查距离值是否为 inf
        if distance > 5:
            img = preprocess(frame1).unsqueeze(0)
            img = img.to(device)
            dis = torch.Tensor(distance).to(device)
            with torch.no_grad:
                error = FSGBM_model(img, dis)
        distance = distance + error

        y1 = y_center - height / 2
        x2 = x_center + width / 2
        text_dis_avg = "dis:%0.2fm" % distance
        cv2.putText(annotated_frame, text_dis_avg,
                    (int(x2 + 5), int(y1 + height + 20)),  # 调整文本位置到框下方
                    cv2.FONT_HERSHEY_SIMPLEX, 1,  # 字体类型和大小
                    (0, 0, 255),  # 红色
                    2,  # 字体厚度为1，变细
                    cv2.LINE_AA)  # 抗锯齿

    cv2.imshow("results", annotated_frame)

    # 检测按键是否按下 'q' 退出
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# 释放摄像头资源
capture.release()
cv2.destroyAllWindows()
# 释放资源
capture.release()

# 关闭所有窗口
cv2.destroyAllWindows()
