from ultralytics import YOLO

model = YOLO("yolov10n.pt")
model.export(format="onnx", opset=13)
model.save("yolo10nano_local.pt")
