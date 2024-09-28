from ultralytics import YOLO 
model = YOLO('best.pt')  # load a pretrained model (recommended for training)


# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model.predict(source='WhatsApp Image 2024-09-28 at 18.19.47_4d1462e1.jpg',conf=0.25,save=True)
print(results)
