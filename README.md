# wearing_mask_or_not_jetsonNano
背景：在nvidia的jetson nano上面跑一个检测是否带口罩的程序
设备详细信息：

![image](https://github.com/lengkujiaai/wearing_mask_or_not_jetsonNano/blob/main/images/1_%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E5%89%8D%E7%9A%84%E6%88%AA%E5%9B%BE.png)

寻找代码，首先找到了：
https://github.com/MiguelOcegueraM/wearingMaskOrNot/blob/master/train_mask_detector02.py
下载后，程序缩进有多处修改，代码有点乱，有多处修改

在readme中看到了，该代码来自印度的一个作者：
https://github.com/prajnasb/observations/tree/master/mask_classifier/Data_Generator

代码的数据集中有不戴口罩的照片686张，在without_mask文件夹中；戴口罩的照片690张，在with_mask文件夹中。

训练全部图片共用时55分钟，训练结束时的with_mask准确率97%，without_mask准确率100%
去掉增强图片后剩余约480张，重新训练这480张图片共用时33分钟

如果加载模型时报错，注意应该在某版本tensorflow下训练的模型就必须在该版本下运行，别人训练的模型在你的tensorflow下可能报错。

运行前需要先安装一些python库：

0、jtop (如果你想看到cup等相关的信息，可以在一个终端中运行jtop)

1、sudo  pip3 install imutils

2、sudo pip3 install sklearn----如果安装不上，卸载enum34后安装成功

3、sudo pip3 uninstall enum34

4、sudo pip3 install sklearn

5、sudo python3 train_mask_detector02.py -d dataset/

这个时候如果运行第5步可能会在几分钟后报错（Resource exhausted: OOM when allocating tensor with shape[10000,32,28,28]）并停止。

原因是虽然你设置了batch_size,但tensorflow默认是一次把所有数据都放进GPU。用CPU训练就可以了，我已经添加了如下几行：

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

再运行第5步就可以了。由于我在代码中已经做了修改，所以你直接运行就可以了。

运行过程见截图：

![image](https://github.com/lengkujiaai/wearing_mask_or_not_jetsonNano/blob/main/images/2_%E8%BF%90%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%97%B6CPU%E7%9A%84%E7%8A%B6%E6%80%81.png)

运行结果：

![image](https://github.com/lengkujiaai/wearing_mask_or_not_jetsonNano/blob/main/images/3_%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E5%90%8E%E7%9A%84%E7%BB%93%E6%9E%9C.png)
运行完程序，会在本文件夹下生成mask_detector.model和plot.png

plot.png记录是训练时损失和准确率的变化：

![image](https://github.com/lengkujiaai/wearing_mask_or_not_jetsonNano/blob/main/images/4_plot.png)

调用摄像头前需要先确认一下摄像头，在终端输入：ls /dev/video*后，如果有两个摄像头会看到类似/dev/video0    /dev/video1,如果有一个摄像头会看到/dev/video0。如果有多个摄像头，假设/dev/video0是CSI摄像头，/dev/video1是USB摄像头，
现在想用usb摄像头，需要把usb_camera_detect()函数下面的vs = VideoStream(src=xx).start()中的xx换成1就行了。注意我代码中并不是写的xx而是对应的摄像头。还需要把if __name__ == “__main__”:下面usb_camera_detect()前面的#号去掉，保存并退出，再运行就可以了。
现在想用CSI摄像头，只需要把if __name__ == “__main__”:下面usb_camera_detect()前面的#号去掉，保存并退出，再运行就可以了。注意，紧邻的另外两个函数前面得有#号。

运行模型，测试一下结果：
sudo python3 shi-detect_mask_video.py -m mask_detector.model

加载时，显示视频的框可能会卡一下，不要慌，稍等一分钟就可以了。

需要注意的是，由于jetson nano有CSI相机接口，也可以使用USB相机，所以我在shi-detect_mask_video.py中有三个函数。其中just_csi_camera()只是调用一下CSI相机，并没有调用检测口罩的模型，运行较快，可以作为检测相机是否正常使用。usb_camera_detect()是调用usb相机检测是否戴口罩，csi_camera_detect()是调用CSI相机检测是否带口罩。



