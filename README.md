# Background Subtraction on SOTA Object Detectors

This repository contains all scripts, configurations, and resources required to reproduce the experiments of the project **“Background Subtraction on SOTA Object Detectors: Does It Worth?”**  
The main goal is to evaluate the impact of combining *Background Subtraction* (BGS) techniques with state-of-the-art object detection models like **YOLOv8** and **Faster R-CNN (Detectron2)**.

---

## Repository Structure

- **scripts/**: Main scripts for comparisons and analytical terms
- **models/**: SOTA models scriptss.
- **main.py**: Generates results between folders with absolute coordinates

---

## Required Libraries

To run this project, make sure to have the following key Python libraries installed:

- `opencv-python`
- `numpy`
- `torch`
- `pathlib` (standard library, but good to check)
- `ultralytics` (for YOLOv8)
- `detectron2` (for Faster R-CNN)

---

## Important Notice

This pipeline critically depends on two major frameworks:

 **Detectron2 (Meta AI)**  
 **YOLO (Ultralytics)**

> **Note:**  
Detectron2 requires proper installation and configuration, including matching your `torch` and `CUDA` versions (if using GPU acceleration). Please follow the [official Detectron2 installation guide](https://detectron2.readthedocs.io/) to ensure compatibility.

> **Note:**  
YOLOv8 (Ultralytics) should be installed via `pip` (`pip install ultralytics`) to ensure you have the latest stable version with full functionality.

---

## Recommended Installation

We strongly recommend using a virtual environment (e.g., `conda` or `venv`) to avoid conflicts.  
Example with **conda** and **CUDA 11.8**:
