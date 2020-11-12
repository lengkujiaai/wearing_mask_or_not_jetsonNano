from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
class detect():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    #调用csi相机需要该函数，可以根据需要更改对应参数
    def gstreamer_pipeline(self, capture_width=1080, capture_height=720, display_width=1080, display_height=720, framerate=30, flip_method=0,):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )
    
        #调用训练好的模型进行检测是否带口罩
    def detect_and_predict_mask(self, frame, faceNet, maskNet, args):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        faces = []
        locs = []
        preds = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                
                faces.append(face)
                locs.append((startX, startY, endX, endY))      
                
        if len(faces) > 0:
            preds = maskNet.predict(faces)
        return (locs, preds)

    #通过模型检测是否带口罩
    def show_detect(self, cap, flag):
        while True:
            if flag == 'usb':
                frame = cap.read()
                #frame = imutils.resize(frame, width=400)
            if flag == 'csi':
                ret,frame = cap.read()
            #frame = imutils.resize(frame, width=400)
            (locs, preds) = self.detect_and_predict_mask(frame, self.faceNet, self.maskNet, self.args)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "mask" if mask > withoutMask else "no mask"
                color = (0, 255, 0) if label == "mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.imshow("detect mask", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        return 0

    #调用usb摄像头检测是否带口罩
    def usb_camera_detect(self):
        print("[INFO] starting video stream...")
        cap = VideoStream(src=1).start()
        time.sleep(2.0)
        self.show_detect(cap,'usb')
        cv2.destroyAllWindows()
        cap.stop()
    
    #调用csi摄像头检测是否带口罩
    def csi_camera_detect(self):
        print("[INFO] starting video stream...")
        cap = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        time.sleep(2.0)

        self.show_detect(cap, 'csi')

        cv2.destroyAllWindows()
        cap.release()

    #循环显示摄像头的图片，直到按下q键后推出显示
    def show_image(self, cap):
        while True:
            ret,frame = cap.read()
            cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    #通过opencv的方法调用csi摄像头，只是显示图片
    def just_csi_camera(self):
        cap = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        self.show_image(cap)
        cap.release()
        cv2.destroyAllWindows()

    #通过VideoStream()的方法调用摄像头，只是显示图片
    #注意，如果没有对应的摄像头，是调不到的，可以通过在终端中输入ls /dev/video* 看是否有video1的显示
    #video1对应函数中VideoStream(1)中的1,如果调用video2，就把对应的参数改成2
    def just_usb_camera(self):
        cap = VideoStream(1).start()
        self.show_image(cap)
        cap.stop()
        cv2.destroyAllWindows()

    #通过opencv的方法调用usb摄像头，只是显示图片
    def just_usb_camera2(self):
        cap = cv2.VideoCapture(1)
        self.show_image(cap)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    d = detect()
    #just_×××函数只是测验一下调用相机是否正常
    #d.just_csi_camera()
    #d.just_usb_camera()
    #d.just_usb_camera2()

    #usb_camera_detect()和csi_camera_detect()是调用不同相机对是否带口罩进行检测
    d.usb_camera_detect()
    #d.csi_camera_detect()
    
    #测验题：
    #使用者可以参考just_usb_camera()和just_usb_camera2()的实现方法，把usb_camera_detect()中的VideoStream().start()改成cv2.VideoCapture()的形式,在自己定义的新函数中实现
    #注意show_detect()中可能也有地方要修改
    #并且把对应的import注释掉
