# YOLO + MiDaS Video Detector 🧠📹

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![MiDaS](https://img.shields.io/badge/MiDaS-Depth%20Estimation-green)](https://github.com/isl-org/MiDaS)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-lightgrey)](https://opencv.org/)
[![Torch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)

## 🎯 Descripción

Este proyecto procesa un video e identifica objetos como **vehículos, personas, bicicletas y perros**, aplicando detección con modelos YOLOv8 y estimación de profundidad con **MiDaS** para calcular distancias aproximadas a la cámara. Ideal para tareas de visión artificial, movilidad inteligente y análisis espacial.

## 🧠 Características

- Detección con dos modelos YOLOv8:
  - Modelo personalizado (`best.pt`) para vehículos.
  - Modelo preentrenado COCO (`yolov8n.pt`) para personas, bicicletas y perros.
- Estimación de profundidad usando MiDaS (`MiDaS_small`).
- Cálculo de distancias estimadas usando:
  - Mapa de profundidad (MiDaS).
  - Tamaño real del objeto (fallback).
- Anotación en video con etiquetas, cajas y distancias.
- Guardado automático del video procesado.

## 📁 Estructura del Proyecto
📂 MiDaSDetector/
├── main.py
├── best.pt
├── GH012372_no_audio.mp4
├── runs_local/
│ └── advanced_video_processing/
│ └── GH012372_no_audio_annotated_depth_distance.mp4
└── requirements.txt

## ⚙️ Requisitos

```bash
pip install -r requirements.txt

Asegúrate de tener instalado el modelo MiDaS_small con torch.hub.



