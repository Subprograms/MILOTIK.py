This Python script, **MILOTIK**, is a sophisticated tool designed for digital forensics and malware analysis, specifically targeting the **Windows Registry**. It uses Machine Learning (ML) to classify registry keys based on their potential for malicious activity, focusing on tactics like **Defense Evasion** and **Persistence**.

---

# MILOTIK: Machine Learning-Based Classification of ATT&CK Tactics and Techniques 

**MILOTIK** is a comprehensive pipeline for parsing Windows Registry hives, engineering security-focused features, and training Ensemble ML models to detect indicators of compromise (IoC).

## Key Features

* **Registry Parsing:** Directly recurses through Windows Registry hives (using `regipy`) to extract structural and metadata features such as key depth, value counts, and subkey distribution.
* **Intelligent Labeling:** Automatically labels datasets by matching registry keys against user-provided malicious and tagged artifact lists.
* **Advanced Feature Engineering:**
* Categorizes paths (e.g., Startup, Service, Network).
* One-hot encoding for Registry types and common malicious key names.
* Structural analysis of key paths for deep-level anomaly detection.


* **Ensemble ML Pipeline:** * Uses **Balanced Random Forest Classifiers** to handle the heavy class imbalance typical in security logs.
* Implements **Recursive Feature Elimination (RFE)** to identify the most significant indicators of malicious intent.
* Optimizes classification thresholds using **ROC AUC** analysis for high-fidelity detection.


* **Interactive GUI:** Built with Tkinter, featuring real-time training metrics, feature importance tabs, and zoomable Decision Tree visualizations.

---

## Installation & Requirements

### Prerequisites

* Python 3.8+
* Windows Registry Hives (for parsing) or existing raw CSV exports.

### Dependencies

Install the required libraries using the provided requirements file:
```bash
pip install -r requirements.txt

```

---

## How to Use

### 1. Data Preparation

1. Launch the application: `python MILOTIK.py`.
2. **Set Hive Path:** Select the Windows Registry hive you wish to analyze.
3. **Set Malicious/Tagged Keys:** Provide `.txt` files containing known malicious registry paths/values for automated labeling.

### 2. The ML Pipeline

1. **Make Dataset:** Click this to parse the hive and generate a preprocessed, labeled CSV ready for training.
2. **Start ML Process:** This initiates the training sequence:
* **SMOTE** is applied to balance the "Benign" vs "Malicious" samples.
* **RFE** prunes the features based on the percentage defined in the GUI.
* **Grid Search** tunes the model for the best performance.



### 3. Analysis & Results

* **Metrics List:** View Accuracy, Precision, Recall, and F1 scores for the Label, Defense, and Persistence models.
* **Feature Tabs:** See which registry attributes (like "Service Path" or "Run Keys") most influenced the model's decision.
* **Tree Visualizer:** Interactively explore the logic of the best-performing decision tree within the forest.

---

## Project Structure

* `MILOTIK.py`: The main application core and GUI logic.
* `preprocessData()`: Logic for transforming raw registry strings into numerical ML features.
* `trainAndEvaluateModels()`: The core training engine utilizing `BalancedRandomForestClassifier`.

---


This project is intended for educational and forensic research purposes.

Would you like me to help you draft a **Technical Documentation** section that explains how the "Path Categorization" logic works for different registry hives?
