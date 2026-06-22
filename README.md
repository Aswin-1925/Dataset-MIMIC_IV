<h1 align="center">🏥 Proactive Rescue: AI-Driven Early Warning System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-1.7.5-orange" alt="XGBoost">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.2.2-yellow" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

<p align="center">
  <strong>An Explainable AI (XAI) pipeline designed to predict non-ICU clinical deterioration 6-12 hours in advance, bridging the hospital monitoring gap.</strong><br>
  Developed by <strong>Aswin VS</strong> (BT5022) | <em>Karunya Institute of Technology and Sciences</em>
</p>

---

## 📑 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [The Solution](#-the-solution)
3. [Dataset & Challenges](#-dataset--challenges)
4. [Technical Architecture](#-technical-architecture)
5. [Key Findings & Explainable AI](#-key-findings--explainable-ai)
6. [How to Run](#-how-to-run)

---

## 🚨 Problem Statement
Hospitals face a dangerous **"monitoring gap"** in general wards, where patients are typically checked by nursing staff only every 4 to 8 hours. Subtle physiological declines—such as early-onset sepsis or respiratory failure—often go unnoticed between these checks, leading to "Failure to Rescue" (preventable death or emergency ICU transfers). Current manual scoring systems are reactive and frequently trigger too late.

## 💡 The Solution
This project develops an AI-powered **"Predictive Smoke Alarm."** By analyzing the last 24 hours of a patient's physiological trends, the model predicts the probability of a life-threatening event 6 to 12 hours before it occurs, providing a critical **"Golden Window"** for clinical intervention.

---

## 📊 Dataset & Challenges
This project utilizes the **MIMIC-IV v2.2 Demo Dataset**, a de-identified electronic health record (EHR) database from the Beth Israel Deaconess Medical Center.

* **Severe Class Imbalance:** 94.5% of patients (260) are stable, while only 5.5% (15) experienced deterioration.
* **Data Sparsity:** Irregular sampling intervals for laboratory tests necessitated Last Observation Carried Forward (LOCF) imputation.
* **The Accuracy Paradox:** Standard models achieved high accuracy by ignoring the minority class (0% Recall). This was mitigated using **SMOTE** (Synthetic Minority Over-sampling Technique).

---

## ⚙️ Technical Architecture

The pipeline processes raw, irregular clinical time-series data, balances the training space, and applies cost-sensitive gradient boosting.

```mermaid
graph TD
    A[MIMIC-IV EHR Data] --> B{Data Preprocessing}
    B -->|LOCF Imputation| C[Handle Missing Labs]
    B -->|Aggregation| D[Mean & Std Volatility]
    C --> E[Data Splitting]
    D --> E
    E -->|80% Train Set| F[SMOTE Balancing]
    F -->|50/50 Class Ratio| G[XGBoost Classifier]
    E -->|20% Test Set| H[Model Evaluation]
    G -->|scale_pos_weight| H
    H --> I[Explainable AI - XAI]
    I -->|Global| J[Feature Importance]
    I -->|Local| K[Clinical Alarm Diagnosis]
    
    classDef primary fill:#1f77b4,stroke:#fff,stroke-width:2px,color:#fff;
    classDef secondary fill:#ff7f0e,stroke:#fff,stroke-width:2px,color:#fff;
    classDef highlight fill:#2ca02c,stroke:#fff,stroke-width:2px,color:#fff;
    
    class G,I highlight;
    class A primary;
    class F secondary;
```
