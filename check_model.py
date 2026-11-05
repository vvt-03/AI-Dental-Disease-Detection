from ultralytics import YOLO

MODEL_PATH = "models/finetuned_800_best.pt"

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print("Task:", model.task)
print("Class names:", model.names)
print("Number of classes:", len(model.names))
print("Stride:", model.model.stride)
