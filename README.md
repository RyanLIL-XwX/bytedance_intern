# ByteDance Internship Projects (Summer 2024, Aug 2024 - Sep 2024)

This repository documents four major computer vision and machine learning tasks completed during my internship at **ByteDance**. Each project demonstrates the application of classical and modern techniques in image processing, object detection, motion analysis, and classification. All implementations are done in **Python**, with frameworks including **OpenCV**, **NumPy**, **PyTorch**, and **scikit-learn**.

---

## 📌 Task 1: Multi-Image Stitching with Homography and Feature Matching

**Description:**  
Implemented a robust image registration and stitching system using SIFT keypoint detection, feature matching, fundamental matrix filtering, and homography-based warping. The pipeline evaluates pairwise stitchability and constructs connected image mosaics using graph traversal and global warping.

**Key Features:**
- SIFT-based feature extraction and BFMatcher with Lowe's ratio test  
- Fundamental matrix & homography estimation using RANSAC  
- Stitchability decision based on geometric consistency  
- Multi-image mosaic construction using BFS over homography graph  
- Final mosaic blending with feathered averaging  

**Tech Stack:** Python, OpenCV, NumPy

---

## 📌 Task 2: Fashion MNIST Classification + Custom Image Inference

**Description:**  
Built a classification pipeline combining PCA-based dimensionality reduction and k-NN classifier for Fashion-MNIST dataset. Extended the project to process and evaluate real-world fashion images using PyTorch and custom dataset preprocessing logic.

**Key Features:**
- Download and preprocess Fashion-MNIST with NumPy  
- Train and evaluate PCA + k-NN classifier (with accuracy curves)  
- Load, normalize, and classify external images with a PyTorch Dataset  
- Visualize classification results with predicted labels  

**Tech Stack:** Python, NumPy, scikit-learn, Matplotlib, PyTorch, PIL

---

## 📌 Task 3: RCNN-Based Object Detection with Evaluation & Visualization

**Description:**  
Developed an RCNN-style object detector using a pretrained ResNet-18 backbone. Implemented classification and bounding box regression heads, trained on both small and large datasets, and evaluated using mAP and per-class recall.

**Key Features:**
- Pretrained ResNet-18 backbone with frozen layers  
- Classification + regression heads with joint loss  
- Post-processing using Non-Maximum Suppression (NMS)  
- Evaluation with mAP and confusion matrix  
- Visualized bounding boxes on test images  

**Performance Highlights:**  
- mAP 0.604 on 4-class dataset  
- mAP 0.492 on 20-class dataset  

**Tech Stack:** Python, PyTorch, Matplotlib

---

## 📌 Task 4: Independent Motion Detection via Optical Flow & FOE Estimation

**Description:**  
Implemented a complete motion analysis pipeline to detect independently moving objects using sparse optical flow. Estimated camera motion via FOE (Focus of Expansion) and separated independently moving points using geometric deviation.

**Key Features:**
- SIFT + Lucas-Kanade optical flow tracking  
- FOE estimation with robust filtering  
- Adaptive parameter tuning with clustering metrics (Silhouette Score)  
- Independent motion detection based on angular deviation from FOE  
- KMeans clustering and bounding box visualization  

**Challenges Solved:**
- Stabilized FOE estimation under noisy conditions  
- Refined threshold tuning strategy using motion quality and clustering consistency  

**Tech Stack:** Python, OpenCV, NumPy, scikit-learn

---

## 🏁 Summary

These four projects provided hands-on experience with key computer vision tasks and advanced techniques in optimization, feature tracking, and model evaluation. Through this internship at **ByteDance**, I strengthened my skills in:

- Designing modular and scalable CV pipelines  
- Debugging numerical instability and convergence issues  
- Evaluating models with both quantitative and qualitative metrics  
- Bridging academic methods and real-world robustness

Each task has standalone scripts and detailed documentation in its respective folder. For detailed implementations, please refer to the corresponding `.py` or `.ipynb` files.

---

**Author:** Hanzhen Qin  
**Internship Period:** Summer 2024 
**Company:** ByteDance  