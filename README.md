# Enhanced Car Detection and Ranging in Adverse Conditions Using Improved YOLO-LiRT and FSGBMN.
## 这是关于利用FSGBMN 和改进的YOLO 进行目标检测测距的代码, 实时进行目标检测和测距
## 环境依赖: ultralytics-YOLO, opencv2, torch, thread
## 使用说明：


Stereo_with_FSGBMN.py 文件为运行系统的代码，YOLO-LiRT.pt是目标检测网络的权重，使用前请下载YOLO源代码从:https://github.com/ultralytics/ultralytics， FSGBMN权重下载:
链接：https://pan.baidu.com/s/1RcI-pRWcfDhqA7wnwoWaLg?pwd=luyh 
提取码：luyh 


由于目标测距使用的是SGBM算法，对于每个双目相机对应的相机内参不一样，如果使用请修改Stereo_with_FSGBMN.py里面的相机内参参数. 对于FSGBMN，由于每个相机内参不一样，如果直接使用可能会造成精度提高不明显，所以可以使用自己的相机采集数据，训练来校正FSGBMN模型，其中Stereo_load_data_with_fsgbm.py 为自动捕捉数据，前提是知道到目标的真实距离，可以利用激光测距等手段获得。Visual_system.py 文件为FSGBMN的训练文件，将下列的代码改成用Stereo_load_data_with_fsgbm.py捕捉的训练数据文件夹，然后开启训练即可获得FSGBMN的训练权重，然后结合Stereo_with_FSGBMN.py使用。 注意也需要修改Visual_system.py 中的相机内参，这样能保证FSGBMN训练的模型适合您的相机。

训练Stereo_load_data_with_fsgbm.py需要修改的代码如下：


training_datasets = FSBGM_Datasets('I:\\visual_system_data\\latastes\\FSGBM\\train\\distance_data.csv',
                                       'I:\\visual_system_data\\latastes\\FSGBM\\train', transform=transform)
    testing_datasets = FSBGM_Datasets('I:\\visual_system_data\\latastes\\FSGBM\\val\\distance_data.csv',
                                      'I:\\visual_system_data\\latastes\\FSGBM\\val', transform=transform)

同时，需要将Stereo_with_FSGBMN.py 和 stereo_load_data_with_fsgbm.py中目标检测的权重位置修改为YOLO-LiRT.pt的位置,修改的代码内容为:

model = YOLO("F:\\algorithm\\ultralytics-main\\yolov8n.pt") 修改为YOLO-LiRT.pt的位置,或您自己的模型的位置。

## 此外，该系统方法不仅仅可以对汽车进行目标检测和测距，对任何物体都可以，只需要修改模型权重为对应的权重即可，以及获取FSGBMN的权重, 以及不使用FSGBMN，不使用FSGBMN请将FSGBMN拟合误差的部分代码删除即可，只有YOLO加SGBMN在光照条件和近距离的情况下效果差强人意，但遇到远距离，>6m 或光照不足会出现精度较差的情况, 此为SGBM算法的通病.论文录取后将拍摄视频介绍详细使用方法。


