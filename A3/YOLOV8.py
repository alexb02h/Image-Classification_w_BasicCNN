import cv2
from ultralytics import YOLO
model = YOLO("runs/detect/train4/weights/best.pt")

video = "GX011160.MP4"
cap = cv2.VideoCapture(video)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_processed.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        result = model.predict(frame, conf=0.5,imgsz=1280,verbose=False)
        annotated_frame = result[0].plot()
        out.write(annotated_frame)
    else: break
    
cap.release()
out.release()
cv2.destroyAllWindows()
