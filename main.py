# OBJECT DETECTOR

import cv2  # OpenCV
import matplotlib.pyplot as plt

configfile = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Config File for Trained MobileNet SSD Model
frozen_model = "frozen_inference_graph.pb"  # Weighhts - derived from tensor flow

model = cv2.dnn_DetectionModel(frozen_model, configfile)  # Model used to Detect Objects

Class_labels = []  # 80 Classes of Classification available in the COCO Dataset Collection
with open("labels.txt", 'rt') as fpt:
    Class_labels = fpt.read().strip("\n").split("\n")  # Extracting the .txt file into List of Labels
    Class_labels.sort()

# print("List of Labels: " + str(Class_labels))
# print("NO OF LABELS: " + str(len(Class_labels)))

# Customizing input format provided to our Model
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

videofile = "C:/Users/gauta/Documents/Videos/Iron Man (Tony Stark) - Hall Of Fame.mp4"
# cap = cv2.VideoCapture(videofile)

# For Webcam
cap = cv2.VideoCapture(0)

# Font options used in labelling the Objects in Frame
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret == True:
        ClassIndex, Confidence, bbox = model.detect(frame, confThreshold=0.55)  # Class Index values start from 1 
        if len(ClassIndex) != 0:  # if Objects Found
            for ClassInd, Confid, Boxes in zip(ClassIndex.flatten(), Confidence.flatten(), bbox):
                if ClassInd >=1 and ClassInd <=80:
                    cv2.rectangle(frame, Boxes, (255, 0, 0), 2)  # Boxes overlayed around the Detected object
                    string = str(Class_labels[ClassInd-1])  # the Label Displayed in the Image
                    cv2.putText(frame, string, (Boxes[0]+10, Boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
        cv2.imshow("OBJECT DETECTION", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()