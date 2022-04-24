# Modified by Augmented Startups & Geeky Bee
# October 2020
# Facial Recognition Attendence GUI
# Full Course - https://augmentedstartups.info/yolov4release
# *-
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer, QDate
from PyQt5.QtWidgets import QDialog, QMessageBox
import cv2
from keras.models import model_from_json  
from keras.preprocessing import image  
import numpy as np
import datetime
import os
import face_recognition as fr

import pandas as pd
import csv


class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        loadUi("./outputwindow.ui", self)
        self.image = None
        self.logger = 1*1000#60*1000 # in seconds
        self.counter = 0
        self.emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

        #Update time
        now = QDate.currentDate()
        current_date = now.toString('ddd dd MMMM yyyy')
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.Date_Label.setText(current_date)
        self.Time_Label.setText(current_time)

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
        self.encode_list = []
        
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
            #faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
           

            boxes = fr.face_locations(test_img)
            encodes_cur_frame = fr.face_encodings(test_img, boxes)[0]
            for box in boxes:
                _, pred_emotion = self.predict_emotion(box, gray_img, test_img, self.model)
                #self.emo_encode_list.append(pred_emotion)
                self.annotate_img(test_img, box, pred_emotion, tgt_fname)
            # encode = face_recognition.face_encodings(img)[0]
            self.encode_list.append(encodes_cur_frame)

        self.timer.timeout.connect(self.update_frame)  # Connect timeout to the output function
        self.timer.start(40)  # emit the timeout() signal at x=40ms

    
    def predict_emotion(self, face_detected, gray_img, img, model):
        
        y1,x2,y2,x1 = face_detected
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),thickness=7)  
        roi_gray=gray_img[y1:y2,x1:x2]#cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img_pixels = image.img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  
    
        preds = model.predict(img_pixels)  
    
        #find max indexed array  
        max_index = np.argmax(preds[0])  
        
        predicted_emotion = self.emotions[max_index]  
        
        return preds, predicted_emotion
            

    def annotate_img(self, img, face_detected, pred_emotion, fname=None):

        x,y,_,_ = face_detected #w,h not needed
        cv2.putText(img, pred_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
        
        resized_img = cv2.resize(img, (1000, 700))  
        if fname==None:
            return resized_img
        else:
            cv2.imwrite(fname,resized_img)


    def mark_emotion(self, name):
            """
            :param name: detected face known or unknown one
            :return:
            """
            with open('Emotion.csv', 'a') as f:
                date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                f.writelines(f'\n{name},{date_time_string}')


    #code to modify
    def face_rec_(self, frame, encode_list_known, class_names):
        """
        :param frame: frame from camera
        :param encode_list_known: known face encoding
        :param class_names: known face names
        :return:
        """
        def mark_emotion(name):
            """
            :param name: detected face known or unknown one
            :return:
            """
            if self.ClockInButton.isChecked():
                self.ClockInButton.setEnabled(False)
                with open('Emotion.csv', 'a') as f:
                        if (name != 'unknown'):
                            buttonReply = QMessageBox.question(self, 'Welcome ' + name, 'Are you Clocking In?' ,
                                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                            if buttonReply == QMessageBox.Yes:

                                date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                                f.writelines(f'\n{name},{date_time_string},Clock In')
                                self.ClockInButton.setChecked(False)

                                self.NameLabel.setText(name)
                                self.StatusLabel.setText('Clocked In')
                                self.HoursLabel.setText('Measuring')
                                self.MinLabel.setText('')

                                #self.CalculateElapse(name)
                                #print('Yes clicked and detected')
                                self.Time1 = datetime.datetime.now()
                                #print(self.Time1)
                                self.ClockInButton.setEnabled(True)
                            else:
                                print('Not clicked.')
                                self.ClockInButton.setEnabled(True)
            elif self.ClockOutButton.isChecked():
                self.ClockOutButton.setEnabled(False)
                with open('Emotion.csv', 'a') as f:
                        if (name != 'unknown'):
                            buttonReply = QMessageBox.question(self, 'Cheers ' + name, 'Are you Clocking Out?',
                                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                            if buttonReply == QMessageBox.Yes:
                                date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                                f.writelines(f'\n{name},{date_time_string},Clock Out')
                                self.ClockOutButton.setChecked(False)

                                self.NameLabel.setText(name)
                                self.StatusLabel.setText('Clocked Out')
                                self.Time2 = datetime.datetime.now()
                                #print(self.Time2)

                                self.ElapseList(name)
                                self.TimeList2.append(datetime.datetime.now())
                                CheckInTime = self.TimeList1[-1]
                                CheckOutTime = self.TimeList2[-1]
                                self.ElapseHours = (CheckOutTime - CheckInTime)
                                self.MinLabel.setText("{:.0f}".format(abs(self.ElapseHours.total_seconds() / 60)%60) + 'm')
                                self.HoursLabel.setText("{:.0f}".format(abs(self.ElapseHours.total_seconds() / 60**2)) + 'h')
                                self.ClockOutButton.setEnabled(True)
                            else:
                                print('Not clicked.')
                                self.ClockOutButton.setEnabled(True)
        
        

        # emotion recognition
        #faces_cur_frame = face_recognition.face_locations(frame)
        gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        #encodes_cur_frame = face_recognition.face_encodings(frame, faces_cur_frame)
        # count = 0
        
        #faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        #_, pred_emotion_list = self.predict_emotion(faces_detected, gray_img, frame, self.model)
        #self.encode_list.append(pred_emotion_list)
        #annotated_img=self.annotate_img(frame, faces_detected, pred_emotion_list)
        
        # face recognition
        faces_cur_frame = fr.face_locations(frame)#bbox
        encodes_cur_frame = fr.face_encodings(frame, faces_cur_frame)#embedding 128d
        # count = 0
        for encodeFace, faceLoc in zip(encodes_cur_frame, faces_cur_frame):
            match = fr.compare_faces(encode_list_known, encodeFace, tolerance=0.50)
            _, pred_emotion = self.predict_emotion(faceLoc, gray_img, frame, self.model)
            face_dis = fr.face_distance(encode_list_known, encodeFace)
            name = "unknown"
            best_match_index = np.argmin(face_dis)
            # print("s",best_match_index)
            if match[best_match_index]:
                name = class_names[best_match_index].upper()
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                annotated_img=self.annotate_img(frame, faceLoc, pred_emotion)
                self.pred_data = [name, pred_emotion]
            mark_emotion(name)

            #mark_emotion(', '.join(pred_emotion_list))
        
        return frame


    def showdialog(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("This is a message box")
        msg.setInformativeText("This is additional information")
        msg.setWindowTitle("MessageBox demo")
        msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)


    def ElapseList(self,name):
        with open('Attendance.csv', "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 2

            Time1 = datetime.datetime.now()
            Time2 = datetime.datetime.now()
            for row in csv_reader:
                for field in row:
                    if field in row:
                        if field == 'Clock In':
                            if row[0] == name:
                                #print(f'\t ROW 0 {row[0]}  ROW 1 {row[1]} ROW2 {row[2]}.')
                                Time1 = (datetime.datetime.strptime(row[1], '%y/%m/%d %H:%M:%S'))
                                self.TimeList1.append(Time1)
                        if field == 'Clock Out':
                            if row[0] == name:
                                #print(f'\t ROW 0 {row[0]}  ROW 1 {row[1]} ROW2 {row[2]}.')
                                Time2 = (datetime.datetime.strptime(row[1], '%y/%m/%d %H:%M:%S'))
                                self.TimeList2.append(Time2)
                                #print(Time2)

    
    def update_frame(self):
        ret, self.image = self.capture.read()
        self.displayImage(self.image, self.encode_list, self.class_names, 1)

        self.counter+=1
        
        if self.counter*40 == self.logger:
            self.counter=0
            #do the logging here

    
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
