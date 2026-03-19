# Trafic: Vietnamese Traffic Sign Detection with YOLO11n and UIB Backbone

This repository presents a traffic sign detection pipeline for Vietnamese real-world traffic sign images using Ultralytics YOLO11n.  
The project includes two main experimental branches:

- **YOLO11n baseline**: fine-tuning the original YOLO11n model on the VR-TSD dataset.
- **YOLO11n + UIB Backbone**: replacing the default backbone with a custom UIB-based lightweight backbone while keeping the detection head unchanged.

---

## 1. Project Overview

The main goal of this repository is to evaluate whether a custom lightweight backbone can improve traffic sign detection performance compared with the standard YOLO11n baseline.

The project is organized into two independent parts:

1. `yolo11n/`  
   Baseline training and evaluation using pretrained YOLO11n weights.

2. `pretrain_backbone_yolo11n/`  
   Modified YOLO11n with a custom UIB backbone integrated into the Ultralytics framework.

---

## 2. Repository Structure

```bash
Trafic/
├── README.md
├── requirements.txt
├── datasets/
├── docs/
├── yolo11n/
├── pretrain_backbone_yolo11n/
└── scripts/