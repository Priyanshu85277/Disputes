import cv2
from ultralytics import YOLO

model = YOLO('best.pt')


video_path = 'Drone Footage.mp4'  # Replace with your video file path

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached or error occurred.")
        break


    results = model.predict(source=frame, conf=0.25, save=False, show=False) 


    for result in results:
        boxes = result.boxes  
        labels = result.names 
        
      
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{labels[cls]} {conf:.2f}"

           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow('YOLO Video Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
