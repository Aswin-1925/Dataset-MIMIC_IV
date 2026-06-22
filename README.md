<div align="center">

# 🏥 Proactive Rescue: AI-Driven Early Warning System 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange?logo=xgboost)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.2-yellow?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

An Explainable AI (XAI) pipeline designed to predict non-ICU clinical deterioration 6–12 hours in advance, bridging the hospital monitoring gap. 

---

## 📑 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [The Solution](#-the-solution)
3. [Dataset & Clinical Challenges](#-dataset--clinical-challenges)
4. [Technical Architecture](#-technical-architecture)
5. [Key Findings & Explainable AI](#-key-findings--explainable-ai)
6. [How to Run](#-how-to-run)
7. [Author](#-author)

---

## 🚨 Problem Statement
Hospitals face a dangerous **"monitoring gap"** in general wards, where patients are checked by nursing staff only every 4 to 8 hours. Subtle physiological declines—such as early-onset sepsis or respiratory failure—often go unnoticed between these checks, leading to "Failure to Rescue" (preventable death or emergency ICU transfers). 

## 💡 The Solution
This project develops an AI-powered **"Predictive Smoke Alarm."** By analyzing the last 24 hours of a patient's physiological trends, the model predicts the probability of a life-threatening event providing a "Golden Window" for proactive clinical intervention.

---

## 📊 Dataset & Clinical Challenges
This project utilizes the **MIMIC-IV v2.2 Demo Dataset**, a de-identified electronic health record (EHR) database.

* **Severe Class Imbalance:** 94.5% of patients are stable, while only 5.5% experienced deterioration.
* **Data Sparsity:** Irregular sampling intervals for laboratory tests necessitated Last Observation Carried Forward (LOCF) imputation.
* **The Accuracy Paradox:** Standard models achieved high accuracy by completely ignoring the minority class. This was resolved using **SMOTE** (Synthetic Minority Over-sampling Technique).

---

## ⚙️ Technical Architecture

The pipeline processes raw clinical time-series data, balances the training space, and applies cost-sensitive gradient boosting.

```mermaid
graph TD
    A[MIMIC-IV EHR Raw Data] --> B{Data Preprocessing}
    B -->|LOCF Imputation| C[Handle Missing Biomarkers]
    B -->|Aggregation| D[Calculate Mean & Std Volatility]
    C --> E[Stratified Train/Test Split]
    D --> E
    E -->|80% Train Set| F[SMOTE Balancing]
    F -->|50/50 Class Ratio| G[XGBoost Classifier]
    E -->|20% Test Set| H[Model Evaluation]
    G -->|scale_pos_weight| H
    H --> I[Explainable AI - XAI]
    I -->|Global Driver Analysis| J[Feature Importance Plot]
    I -->|Local Diagnosis| K[Patient-Specific Alarm Cause]
    
    classDef primary fill:#2b3137,stroke:#fff,stroke-width:2px,color:#fff;
    classDef highlight fill:#0052cc,stroke:#fff,stroke-width:2px,color:#fff;
    classDef outcome fill:#128a3c,stroke:#fff,stroke-width:2px,color:#fff;
```

    class A,E,H primary;
    class G,F highlight;
    class J,K outcome;
