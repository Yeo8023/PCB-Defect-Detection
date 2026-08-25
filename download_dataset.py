import os
from roboflow import Roboflow

print("Downloading dataset from Roboflow...")
rf = Roboflow(api_key="9OvQ198Ng7Ud5JKNcdze")
project = rf.workspace("university-2xdiy").project("pcb-defects-chi1b")
version = project.version(6)
dataset = version.download("yolov8")

print(f"Dataset downloaded successfully to: {dataset.location}")
