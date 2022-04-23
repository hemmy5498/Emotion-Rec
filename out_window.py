# Modified by Augmented Startups & Geeky Bee
# October 2020
# Facial Recognition Attendence GUI
# Full Course - https://augmentedstartups.info/yolov4release
# *-
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QDialog
import cv2
from keras.models import model_from_json  
from keras.preprocessing import image  
import numpy as np
import datetime
import os


class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        loadUi("./outputwindow.ui", self)
        self.image = None
        self.emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    @pyqtSlot()
    def startVideo(self, camera_name):
        """
        :param camera_name: link of camera or usb camera
        :return:
        """
        
        #load model  
        self.model = model_from_json(open("fer.json", "r").read())  
        #load weights  
        self.model.load_weights('fer.h5')  
        self.face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  


        if len(camera_name) == 1:
            self.capture = cv2.VideoCapture(int(camera_name))
        else:
            self.capture = cv2.VideoCapture(camera_name)
        self.timer = QTimer(self)  # Create Timer
        path = 'ImagesAttendance'
        tgt_path = 'AnnotatedImagesAttendance'
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(tgt_path):
            os.mkdir(tgt_path)
        # known face encoding and known face name list
        images = []
        self.class_names = []
        self.encode_list = []#maybe mod to emotions detected instead
        attendance_list = os.listdir(path)
        
        # print(attendance_list)
        tgt_fnames = []
        for cl in attendance_list:
            cur_img = cv2.imread(f'{path}/{cl}')
            tgt_fnames.append(f'{tgt_path}/{cl}')
            images.append(cur_img)
            self.class_names.append(os.path.splitext(cl)[0])
        
        for tgt_fname, test_img in zip(tgt_fnames, images):
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            
            #boxes = face_recognition.face_locations(img)# this returns list of tuples of found face locations in css (top, right, bottom, left) order
            faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
            _, pred_emotion_list = self.predict_emotion(faces_detected, gray_img, test_img, self.model)
            self.encode_list.append(pred_emotion_list)
            self.annotate_img(test_img, faces_detected, pred_emotion_list, tgt_fname)

        self.timer.timeout.connect(self.update_frame)  # Connect timeout to the output function
        self.timer.start(40)  # emit the timeout() signal at x=40ms

    
    def predict_emotion(self, faces_detected, gray_img, img, model):
        preds_list = []
        pred_emotion_list = []
        for (x,y,w,h) in faces_detected:  
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
            roi_gray=cv2.resize(roi_gray,(48,48))  
            img_pixels = image.img_to_array(roi_gray)  
            img_pixels = np.expand_dims(img_pixels, axis = 0)  
            img_pixels /= 255  
    
            preds = model.predict(img_pixels)  
    
            #find max indexed array  
            max_index = np.argmax(preds[0])  
        
            predicted_emotion = self.emotions[max_index]  
            
            preds_list.append(preds)
            pred_emotion_list.append(predicted_emotion)
        
        return preds_list, pred_emotion_list
            

    def annotate_img(self, img, faces_detected, pred_emotion_list, fname=None):

        for predicted_emotion, (x,y,_,_) in zip(pred_emotion_list, faces_detected): #w,h not needed
            cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
        
        resized_img = cv2.resize(img, (1000, 700))  
        if fname==None:
            return resized_img
        else:
            cv2.imwrite(fname,resized_img)


    #code to modify
    def face_rec_(self, frame, encode_list_known, class_names):
        """
        :param frame: frame from camera
        :param encode_list_known: known face encoding
        :param class_names: known face names
        :return:
        """
        # csv
        def mark_emotion(name):
            """
            :param name: detected face known or unknown one
            :return:
            """
            with open('Emotion.csv', 'a') as f:
                date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                f.writelines(f'\n{name},{date_time_string}')
        

        # face recognition
        #faces_cur_frame = face_recognition.face_locations(frame)
        gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        #encodes_cur_frame = face_recognition.face_encodings(frame, faces_cur_frame)
        # count = 0
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        _, pred_emotion_list = self.predict_emotion(faces_detected, gray_img, frame, self.model)
        self.encode_list.append(pred_emotion_list)
        annotated_img=self.annotate_img(frame, faces_detected, pred_emotion_list)
        mark_emotion(', '.join(pred_emotion_list))
        
        return frame

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.displayImage(self.image, self.encode_list, self.class_names, 1)

    def displayImage(self, image, encode_list, class_names, window=1):
        """
        :param image: frame from camera
        :param encode_list: known face encoding list
        :param class_names: known face names
        :param window: number of window
        :return:
        """
        image = cv2.resize(image, (640, 480))
        try:
            image = self.face_rec_(image, encode_list, class_names)
        except Exception as e:
            print(e)
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)
