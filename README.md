# PCB Defect Detection System

An automated computer vision system designed to detect manufacturing defects on Printed Circuit Boards (PCBs). This project leverages state-of-the-art **YOLOv8** object detection to identify anomalies such as missing holes, mouse bites, open circuits, shorts, spurs, and spurious copper in real-time.

## 🎯 Project Overview
Quality assurance in PCB manufacturing is a critical bottleneck. Manual inspection is slow and prone to human error. This project demonstrates an automated AI-driven approach to detect surface defects on bare PCBs, significantly improving inspection speed and reliability.

## 🚀 Key Features
- **Custom Trained YOLOv8 Model**: Fine-tuned specifically on industrial PCB datasets.
- **Multi-Class Defect Detection**: Capable of identifying 6 distinct types of PCB manufacturing flaws.
- **High-Speed Inference**: Built for real-time edge deployment.

## 🛠️ Technologies Used
- **Python 3**
- **YOLOv8 (Ultralytics)**: Core object detection framework.
- **PyTorch**: Deep learning backend optimized for GPU acceleration.
- **Roboflow**: Used for dataset annotation, formatting, and version control.

## 📂 Dataset
The model was trained on a custom dataset containing images of bare PCBs. The dataset includes annotations for the following defect classes:
1. `missing_hole`
2. `mouse_bite`
3. `open_circuit`
4. `short`
5. `spur`
6. `spurious_copper`

*(Note: The raw dataset and model weights are not included in this repository due to size constraints. The model can be retrained using the provided scripts).*

## 💻 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/PCB-Defect-Detection.git
   cd PCB-Defect-Detection
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install ultralytics
   ```

## 🧠 Training the Model
To train the model from scratch, ensure you have downloaded a YOLO-formatted dataset, updated the path in `train.py`, and run:
```bash
python train.py
```
*Training is configured to run on a dedicated GPU (e.g., RTX 5040) for significantly faster processing times.*

## 📈 Future Improvements
- [ ] Develop a Streamlit/Gradio web interface for easy drag-and-drop image testing.
- [ ] Deploy the model as a REST API using FastAPI.
- [ ] Export the model to ONNX or TensorRT for optimized edge inference.

---
*This project was developed to demonstrate practical skills in Computer Vision, Data Annotation, and Model Training for industrial applications.*
