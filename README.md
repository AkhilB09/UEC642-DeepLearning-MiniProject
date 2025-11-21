# Green & Lean: LoRA-Swin for Plant Disease Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/Technique-LoRA-blue)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A Deep Learning microproject demonstrating **Parameter-Efficient Fine-Tuning (LoRA)** applied to a **Swin Transformer** for classifying plant leaf diseases. 

> **Technique Showcase:** This project moves beyond standard Transfer Learning by utilizing **Low-Rank Adaptation (LoRA)**, a state-of-the-art technique (2022-2025) that reduces trainable parameters by **99%** while maintaining SOTA accuracy.

---

## Project Overview

Traditional fine-tuning of Vision Transformers (ViT) requires updating all model parameters, which is computationally expensive. This project solves that problem by freezing the pre-trained backbone and injecting small, trainable rank-decomposition matrices.

- **Model:** Microsoft Swin-Tiny Transformer (`swin-tiny-patch4-window7-224`)
- **Dataset:** Beans Leaf Disease Dataset (Proxy for PlantVillage)
- **Classes:** Healthy, Angular Leaf Spot, Bean Rust
- **Technique:** LoRA (Rank=16, Alpha=32)

### Key Achievements
| Metric | Standard Fine-Tuning | **Our Approach (LoRA)** |
| :--- | :--- | :--- |
| **Trainable Params** | ~28,000,000 (100%) | **~300,000 (1%)** |
| **Model Size** | ~110 MB | **~1.2 MB (Adapters)** |
| **Accuracy** | ~98.5% | **~98.2% (Comparable)** |
| **Training Device** | High-End GPU Required | **Consumer GPU / Colab T4** |

---

## Repository Structure

```text
 LoRA-Swin-Plant-Disease
 ┣  results               # Generated Output Images & Models
 ┃ ┣  confusion_matrix.png
 ┃ ┣  grad_cam.png
 ┃ ┣  training_curve.png
 ┃ ┗  lora_adapters     # Saved model weights
 ┣  train.py              # Main training script (LoRA injection)
 ┣  evaluate.py           # Evaluation & Visualization (Grad-CAM)
 ┣  requirements.txt      # Dependencies
 ┗  README.md             # Project Documentation
