# PCB Defect Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-FF9900?style=flat-square&logo=yolo)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-6706CE?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)

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

*(Note: The raw dataset and model weights are not included in this repository due to size constraints. The original dataset can be found on [Roboflow Universe - PCB Defects](https://universe.roboflow.com/university-2xdiy/pcb-defects-chi1b). The model can be retrained using the provided scripts).*

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
*Training is configured to run on a dedicated GPU (e.g., RTX GPU) for significantly faster processing times.*

---
*This project was developed to demonstrate practical skills in Computer Vision, Data Annotation, and Model Training for industrial applications.*
