# # import cv2

# # #read img
# # image = cv2.imread('img.jpg')

# # #convert into gray
# # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # #display img
# # cv2.imshow("gray image", gray_img)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # 
# # -----------------------------------------------<------------------------traffic deetection model----------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.

from ultralytics import YOLO
import torch
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("yolov8n.pt").to(device)

cap = cv2.VideoCapture("trafficVid.mp4")

frame_width = 640
frame_height = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.resize(frame, (frame_width, frame_height))

    results = model(frame, device=device)  
    vehicle_count = len(results[0].boxes)

    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()






# --------------------------------------------<<<face detection -------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>.

import cv2 as cv
import numpy as np
import dlib
from imutils import face_utils
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0,0,0)
def dist(a, b):
    dst = np.linalg.norm(a-b)
    return dst
def blinked(a,b,c,d,e,f):
    up = dist(b, d) + dist(c,e)
    down = dist(a, f)
    ratio = up/(2.0*down)

    if ratio>0.3:
        return 2
    elif 0.3 >= ratio > 0.15:
        return 1
    else:
        return 0
cap = cv.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        face_frame = frame.copy()
        cv.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        left_blink = blinked(landmarks[36],landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        if left_blink==0 and right_blink == 0:
            sleep+=1
            drowsy = 0
            active = 0
            if sleep>6:
                status = "SLEEPING!"
                color = (255, 0, 0)
        
        elif left_blink==1 and right_blink==1:
            sleep = 0
            active = 0
            drowsy+=1
            if drowsy>6:
                status = "DROWSY!"
                color = (0,0,255)
        
        else:
            drowsy = 0
            sleep = 0
            active+=1
            if active>6:
                status = "ACTIVE!"
                color = (0, 255, 0)
        
        cv.putText(frame, status, (100,100), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
        
        cv.imshow("Frame", frame)
        cv.imshow("Result of detector", face_frame)
        key = cv.waitKey(1)
        if key == 27:
            break
