import cv2
import numpy as np
import mediapipe as mp
import tensorflow_hub as hub
from keras.models import load_model

cap = cv2.VideoCapture(0)
mpFacedetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDetection = mpFacedetection.FaceDetection(0.75)
# hand
mpHands = mp.solutions.hands
hands = mpHands.Hands()
#load model
my_model = load_model("Face.h5", custom_objects={'KerasLayer':hub.KerasLayer})

class_name = ["Glasses","Mask","NoGlasses"]
while True:
    ret, frame = cap.read()
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = FaceDetection.process(imgRGB)
    if result.detections:
        for id,detection in enumerate(result.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = frame.shape
            bbox = int(bboxC.xmin * iw),int(bboxC.ymin * ih),\
                    int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame,bbox,(0,225,0),2)
            newimg = frame[int(bboxC.ymin * ih):int(bboxC.ymin * ih) + int(bboxC.ymin * ih),
                     int(bboxC.xmin * iw): int(bboxC.xmin * iw) + int(bboxC.height * ih)]
            newimg = cv2.resize(newimg, dsize=None,fx=0.5,fy=0.5 )
            img_real = newimg.copy()
            img_real = cv2.resize(img_real, dsize = (224,224))
            img_real = img_real.astype('float')*1./255
            img_real = np.expand_dims(img_real,axis=0)
            predict = my_model.predict(img_real)
            print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
            print(np.max(predict[0], axis=0))
            if (np.max(predict) >= 0.7) and (np.argmax(predict[0]) != 0):
                # Show image
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1.5
                color = (0, 255, 0)
                thickness = 2

                cv2.putText(frame, class_name[np.argmax(predict)], org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
    #draw hand
    result2 = hands.process(imgRGB)
    if result2.multi_hand_landmarks:
        for handLms in result2.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)


    if cv2.waitKey(10) & 0XFF == ord('q'):
        break;
    cv2.imshow("Camera",frame)
    cv2.waitKey(10)
cap.release()
cv2.destropyAllWindows()