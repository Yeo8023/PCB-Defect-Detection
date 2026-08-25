from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8n model (the nano version is great for fast training)
    model = YOLO("yolov8n.pt")
    
    print("Starting YOLOv8 training on PCB Defects dataset...")
    
    # Train the model
    # Note: device=0 tells it to use your RTX GPU!
    results = model.train(
        data="d:/Project/Computer Vision/PCB-Defect-Detection/PCB-Defects--6/data.yaml",
        epochs=20,  # Starting with 20 epochs so it doesn't take hours
        imgsz=640,
        batch=16,
        device=0,
        name="pcb_defect_model"
    )
    
    print("Training complete! Your new model is saved to runs/detect/pcb_defect_model/weights/best.pt")

if __name__ == '__main__':
    main()
