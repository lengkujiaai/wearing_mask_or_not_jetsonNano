# wearing_mask_or_not_jetsonNano
背景：在nvidia的jetson nano上面跑一个检测是否带口罩的程序

我的邮箱：lengkujiaai@126.com

jetson nano 软件版本：

![image](https://github.com/lengkujiaai/wearing_mask_or_not_jetsonNano/blob/main/images/0_%E7%89%88%E6%9C%AC%E4%BF%A1%E6%81%AF.png)

训练模型前处理器的状态：

![image](https://github.com/lengkujiaai/wearing_mask_or_not_jetsonNano/blob/main/images/1_%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E5%89%8D%E7%9A%84%E6%88%AA%E5%9B%BE.png)

寻找代码，首先找到了：

https://github.com/MiguelOcegueraM/wearingMaskOrNot/blob/master/train_mask_detector02.py

下载后，程序缩进有多处错误，进行了修改；代码有点乱，有修改；代码有错误，有修改；增加注释，调整代码的组织方式；增加代码功能

在readme中看到了，该代码来自印度的一个作者：

https://github.com/prajnasb/observations/tree/master/mask_classifier/Data_Generator

代码的数据集中有不戴口罩的照片686张，在without_mask文件夹中；戴口罩的照片690张，在with_mask文件夹中。

训练全部图片共用时55分钟，训练结束时的with_mask准确率97%，without_mask准确率100%

去掉增强图片后剩余约480张，重新训练这480张图片共用时33分钟

如果加载模型时报错，注意应该在某版本tensorflow下训练的模型就必须在该版本下运行，别人训练的模型在你的tensorflow下可能报错。

nvidia官方也有一个检测口罩的示例：

https://github.com/NVIDIA-AI-IOT/face-mask-detection

运行前需要先安装一些python库：

0、jtop (如果你想看到cup等相关的信息，可以在一个终端中运行jtop)

1、sudo  pip3 install imutils

2、sudo pip3 install sklearn----如果安装不上sklearn，卸载enum34后安装成功

3、sudo pip3 uninstall enum34-----卸载enum34

4、sudo pip3 install sklearn-----再次安装sklearn

5、sudo python3 shi_train_mask_detector.py -d dataset/

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

plot.png记录的是训练时损失和准确率的变化：

![image](https://github.com/lengkujiaai/wearing_mask_or_not_jetsonNano/blob/main/images/4_plot.png)

调用摄像头前需要先确认一下摄像头，在终端输入：

ls /dev/video*

后，如果有两个摄像头会看到类似

/dev/video0    /dev/video1

如果有一个摄像头会看到/dev/video0。如果有多个摄像头，假设/dev/video0是CSI摄像头，/dev/video1是USB摄像头

现在想用usb摄像头，需要把usb_camera_detect()函数下面的vs = VideoStream(src=xx).start()中的xx换成1就行了

注意我代码中并不是写的xx而是对应的摄像头

还需要把

if __name__ == “__main__”:

下面usb_camera_detect()前面的#号去掉，保存并退出，再运行就可以了

现在想用CSI摄像头，只需要把

if __name__ == “__main__”:

下面csi_camera_detect()前面的#号去掉，保存并退出，再运行就可以了。注意，紧邻的另外两个函数前面得有#号。

运行模型，测试一下结果：

sudo python3 shi_detect_mask_video.py -m mask_detector.model

加载时，显示视频的框可能会卡一下，不要慌，稍等一分钟就可以了。

运行效果：
![image](https://github.com/lengkujiaai/wearing_mask_or_not_jetsonNano/blob/main/images/recorder_csi_detect_short.gif)

在jetson nano上录屏用的是simplescreenrecorder软件，安装：sudo apt-get install simplescreenrecorder

在win10上进行视频转gif是用的：https://www.cockos.com/licecap/

需要注意的是，由于jetson nano有CSI相机接口，也可以使用USB相机，所以我在shi_detect_mask_video.py中写了几个函数。

其中just_csi_camera()只是调用一下CSI相机，并没有调用检测口罩的模型，运行较快，可以作为检测相机是否正常使用。

usb_camera_detect()是调用usb相机检测是否戴口罩，csi_camera_detect()是调用CSI相机检测是否带口罩。

进阶练习：

1、shi_detect_mask_video.py文件最后增加了一个可以修改的小测验，感兴趣的可以修改（代码随时修改，不能保证本readme是最新的）


2、使用者可以自己采集没有带口罩的图片，用作图工具给没戴口罩的图片添加口罩后，分别放到dataset目录下的with_mask和without_mask文件夹中，训练自己的图片看看效果。记得把原有的图片删除，采集样本也是学习神经网络的重要一环呢。如果样本有错误，后面的一切都无从谈起。也可以保留原有的图片，在dataset目录下创建my_with_mask和my_without_mask文件夹来存放自己的样本，但记得修改训练代码的相关路径



另：

1、2020-11-12，现供职于北京中电科卫星导航系统有限公司，本部门为研发中心。

2、公司在淘宝销售nvidia jetson 系列的产品，包括jetson nano，     TX1,     TX2,    AGX XAVIER,        XAVIER NX产品

3、我们属于提供技术支持的，本项目就是一老师要求的功能。

4、复制链接：   

    2.0fυィ直信息₰gyi7clU3sNj₤回t~bao或點几url链 https://m.tb.cn/h.4WAPC9j?sm=19844c 至浏览er【北京中电科卫星导航公司】
    
后打开淘宝即可
