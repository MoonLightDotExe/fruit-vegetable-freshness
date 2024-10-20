from ultralytics import YOLO

model = YOLO('datasets/YOLO_Resources/yolo11n.pt')

model.train(
    data='./data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

metrics = model.val()

print(metrics)

# results = model.predict('path_to_image.jpg')
# results.show()
