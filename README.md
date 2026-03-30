# 🚀 Real-Time Biosignal MLOps Platform

### Production-Grade ML System with Feature Store, Streaming Pipeline, Online Learning & Cloud-Ready Deployment

---

## 📌 Project Overview

This project implements a **production-grade, end-to-end MLOps system** for real-time biosignal processing, designed to simulate an **ICU monitoring pipeline**.

It ingests raw physiological signals (ECG, SpO2, IMU), performs real-time feature engineering, runs lightweight ML inference, detects data drift, and continuously retrains models — all exposed via scalable APIs and an interactive frontend.

> ⚡ Designed with real-world production constraints: low latency, model drift, incremental learning, and modular microservices.

---

## 🎯 Problem Statement

Healthcare monitoring systems require:

* Real-time signal processing
* Accurate predictions under changing patient conditions
* Continuous model adaptation
* Scalable and reliable infrastructure

This system solves these by combining **ML + Data Engineering + MLOps + Full Stack UI**.

---

## 🏗 System Architecture

```text
[ Tablet Device (.bin) ]
            ↓
[ FastAPI Ingestion Layer ]
            ↓
[ Signal Processing + Feature Extraction ]
            ↓
[ Feature Store (Feast-style Repository) ]
            ↓
[ TFLite Model Inference Engine ]
            ↓
[ Drift Detection + Accuracy Monitoring ]
            ↓
[ Conditional Retraining Pipeline ]
            ↓
[ Updated Model + Scaler Artifacts ]
            ↓
[ React Dashboard (Live Monitoring UI) ]
```

---

## ⚙️ Tech Stack

### 🔹 Backend & ML

* Python, FastAPI
* TensorFlow / TFLite
* Scikit-learn

### 🔹 Data Processing

* NumPy, Pandas
* Custom biosignal feature engineering

### 🔹 MLOps & Orchestration

* Drift detection algorithms
* Incremental retraining pipeline
* MLflow (experiment tracking)
* Apache Airflow (workflow orchestration)

### 🔹 Feature Store

* Custom Feast-style feature repository

### 🔹 Frontend

* React.js (real-time dashboard UI)

### 🔹 DevOps

* Docker (containerization)
* Azure (cloud deployment-ready)

---

## 🔄 End-to-End Pipeline

### 1. Data Ingestion

* Accepts raw `.bin` biosignal data (single patient, ≤1 min)

### 2. Signal Processing

* Converts binary → structured format
* Cleans and normalizes signals

### 3. Feature Engineering

* Extracts:

  * Heart rate (BPM)
  * SpO2 levels
  * IMU statistics
* Applies **dynamic baseline correction**

### 4. Feature Store

* Stores structured features for reuse (training + inference)

### 5. Model Inference

* Uses optimized **TFLite model**
* Returns class predictions:

  * Normal (0)
  * Alert (1)
  * Critical (2)

### 6. Drift Detection

* Monitors:

  * Feature distribution shift
  * Prediction mismatch
* Computes drift metrics

### 7. Continuous Learning

* Triggers retraining if drift detected
* Supports:

  * Incremental fine-tuning
  * Class balancing
  * Noise injection

### 8. Model Update

* Exports:

  * `latest_model.tflite`
  * `scaler.json`

### 9. Visualization

* Displays results in React-based ICU dashboard

---

## 🚀 Key Features

* ⚡ Real-time biosignal processing pipeline
* 🧠 Online learning with automatic retraining
* 📉 Drift detection & monitoring
* 🧬 Dynamic baseline computation
* 📦 Lightweight inference using TFLite
* 🧩 Modular microservice architecture
* 🌐 Full-stack system (FastAPI + React)
* ☁️ Cloud-ready deployment design

---

## 📁 Project Structure

```text
biosignal_mlops/
│
├── app/
│   ├── routes/        # API endpoints
│   ├── services/      # core logic (inference, retrain, drift)
│   ├── utils/         # signal + helper functions
│   └── main.py
│
├── biosignal_feature_repo/
│   └── feature_repo/  # feature definitions (Feast-style)
│
├── frontend/          # React UI
│
├── requirements.txt
├── requirements-dev.txt
├── test_drift.py
├── check_db.py
└── check_feature.py
```

---

## ⚡ Local Setup

### 1. Clone repo

```bash
git clone <repo-url>
cd biosignal_mlops
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run backend

```bash
uvicorn app.main:app --reload
```

### 5. Run frontend

```bash
cd frontend
npm install
npm start
```

---

## 📊 Model Lifecycle

* Initial offline training
* Real-time inference
* Drift detection
* Conditional retraining
* Model version update

---

## ☁️ Deployment Strategy

* Dockerized microservices
* Azure App Service / AKS
* CI/CD via GitHub Actions
* Scalable API deployment

---

## 🔮 Future Enhancements

* Kafka-based real-time streaming
* Prometheus + Grafana monitoring
* Feature store (Feast + Redis online store)
* Multi-patient support
* Model registry & versioning

---

## 👩‍💻 Author

**Nandini Arjunan**

---

## ⭐ Show Your Support

If you found this useful, consider giving it a ⭐ on GitHub!
