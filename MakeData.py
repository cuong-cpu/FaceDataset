import cv2
import mediapipe as mp
import os
cam = cv2.VideoCapture(0)
i = 0
mpFacedetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDetection= mpFacedetection.FaceDetection(0.75)

label = "Mask"
while True:
    ret, frame = cam.read()
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = FaceDetection.process(imgRGB)
    # print(result)
    i += 1
    if result.detections:
        for id,detection in enumerate(result.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = frame.shape
            bbox = int(bboxC.xmin * iw),int(bboxC.ymin * ih),\
                    int(bboxC.width * iw), int(bboxC.height * ih)
            # cv2.rectangle(frame,bbox,(0,225,0),2)
            if i > 60 and i<110:
                newimg = frame[int(bboxC.ymin * ih):int(bboxC.ymin * ih)+int(bboxC.ymin * ih), int(bboxC.xmin * iw): int(bboxC.xmin * iw)+int(bboxC.height * ih)]
                print("Img:"+str(i-60))
                if not os.path.exists('Data/' +str(label)):
                    os.mkdir('Data/' + str(label))
                cv2.imwrite('Data/' + str(label) + "/" + str(i) + ".png",newimg)


    cv2.imshow("Camera",frame)
    cv2.waitKey(10)
cam.release()
cv2.destroyAllWindows()
