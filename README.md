# AI-Driven Optimization of Cloud-Hosted Resources 🚀

This repository contains the source code, experimental framework, and predictive models evaluated as part of our Bachelor's Thesis: **AI-Driven Optimization of Cloud-hosted Resources**.

The project investigates proactive resource scaling for cloud environments (CPU and Memory utilization) using machine learning and deep learning architectures to prevent over- and under-provisioning.

---

## 📌 Project Overview

Traditional cloud resource scaling relies heavily on reactive threshold-based rules, which often lead to performance degradation during sudden traffic spikes or resource waste during low-demand periods. 

This research proposes an **AI-driven proactive framework** that predicts future cloud resource utilization using historical telemetry data.

### Key Objectives:
* Compare classical ML models, ensemble methods, and deep learning architectures on real-world and synthetic cloud workload datasets.
* Evaluate dynamic data synthesis (using GANs) for cloud workload augmentation.
* Identify the most efficient predictive architecture for real-time proactive scaling.

---

## 📊 Benchmarked Models

The evaluation benchmarks several machine learning and deep learning models across metrics including **RMSE**, **MAE**, and **$R^2$**:

1. **Classical Machine Learning:**
   * **SVR (Support Vector Regression)** — *Top Performer (Lowest RMSE/MAE & $R^2 = 1.00$)*
2. **Ensemble Methods:**
   * **CatBoost**, **XGBoost**, **Random Forest**
3. **Deep Learning & Hybrid Architectures:**
   * **Autoencoders**
   * **LSTM (Long Short-Term Memory)**
   * **CNN-LSTM (Hybrid Model)**
   * **CNN (Convolutional Neural Network)**
   * **GRU (Gated Recurrent Unit)**

---

## 📁 Datasets Evaluated

The framework evaluates performance across **four key datasets**:
* **DB01, DB02, DB03:** Real-world cloud infrastructure performance telemetry.
* **GAN Dataset:** Synthetic cloud workload data generated via Generative Adversarial Networks (GAN) to test model generalization under augmented scenarios.

---

## 📈 Key Findings & Performance Highlights

* **Support Vector Regression (SVR)** achieved the best predictive accuracy overall, demonstrating exceptional stability against non-linear, noisy, and volatile workload patterns:
  * **CPU Utilization:** Average RMSE ~ `0.92%` | Average MAE ~ `0.785%`
  * **Memory Utilization:** Average RMSE ~ `0.975%` | Average MAE ~ `0.83%`
* **Tree-based Ensembles & Deep Learning:** CatBoost, XGBoost, Autoencoders, and LSTMs formed a strong second tier ($R^2 \approx 0.9975$).
* **Data Augmentation:** GAN-generated datasets proved highly effective in synthesizing realistic cloud workload characteristics for training robust models.
