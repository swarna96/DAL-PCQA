# DAL-PCQA

**DAL-PCQA** (Distortion-Aware Language-Annotated Point Cloud Quality Assessment Dataset) provides structured distortion annotations and natural language quality descriptions for point cloud quality assessment.

Unlike traditional PCQA datasets that only provide Mean Opinion Scores (MOS), DAL-PCQA includes:

- Multi-level distortion severity labels  
- 3D-specific geometric and structural distortion annotations  
- 5-level quality labels (Excellent, Good, Fair, Poor, Bad)  
- Structured natural language descriptions  

The dataset is designed to support language-driven, multimodal, and explainable PCQA research.

---

## Contents

This repository contains:
dal_pcqa_annotations.csv


Each row includes:
- Point cloud identifier  
- Distortion severity levels  
- Original MOS  
- Rounded MOS  
- Discrete quality label  
- Structured natural language description  

---

## Important

This repository **does not include the original point cloud files**.

To obtain the point clouds, please download them from the official repositories:



-  SJTU-PCQA: [Paper](https://ieeexplore.ieee.org/document/9238424) [Dataset](https://vision.nju.edu.cn/28/fd/c29466a469245/page.htm)
- WPC Dataset: [Paper](https://ieeexplore.ieee.org/document/9756929)([Dataset](https://drive.google.com/drive/folders/1dHDqKXgvkUhQdUzT7pJjrJ7zRnceFIkO)

