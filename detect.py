import sys
import cv2 
import imutils
from yolo_trt import YoloTRT

# use path for library and engine file
model = YoloTRT(library="libmyplugins.so", engine="yolov7-tiny-rep-best.engine", conf=0.5, yolo_ver="v7")

cap = cv2.VideoCapture("output1.mp4")

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=416)
    detections, t = model.Inference(frame)
    # for obj in detections:
    #    print(obj['class'], obj['conf'], obj['box'])
    # print("FPS: {} sec".format(1/t))
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()