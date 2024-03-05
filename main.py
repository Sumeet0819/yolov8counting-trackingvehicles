import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt')

cap=cv2.VideoCapture(0)


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()

cy1=322
cy2=368
offset=6

while True:    
    ret,frame = cap.read()
    if not ret:
        count = +1
        break
    frame=cv2.resize(frame,(1080,720))
   

    results=model.predict(frame, agnostic_nms=True)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.rectangle(frame, (x3, y3), (x4 , y4),(249, 34, 108), 2)
        cv2.rectangle(frame, (x3, y3), (x3+80, y3-30),(249, 34, 108), -1)
        cv2.putText(frame,format(c)+""+ str(id),(x3,y3-10),cv2.QT_FONT_NORMAL,0.5,(225, 255,225),2)
           


#    cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1)
#    cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(0)==27:
        break
cap.release()
cv2.destroyAllWindows()

