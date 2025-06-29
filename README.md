# ⚡ Hourly Electricity Demand Batch Prediction Service

## 📌 Project Overview
This project presents a **production-grade, end-to-end ML system** to forecast hourly electricity demand for a **Public Energy Company**, helping optimize energy supply, cut costs, and reduce CO₂ emissions. 

The solution leverages **MLOps best practices** — from automated data ingestion and feature engineering to training, batch inference, and monitoring.

<div align="center">
  <img src="display.gif" alt="Project Demo" width="600"/>
</div>

---

## 🏢 Business Problem

The Public Energy Company faces key operational issues:

- ⚠️ **Overproduction**: Wastes energy and increases costs.
- 🔌 **Shortages**: Leads to outages and unhappy customers.
- 🧃 **Resource Misallocation**: Impacts operational efficiency and raises carbon footprint.

These challenges drive up expenses, hurt profitability, and lower customer satisfaction.

---

## 🧠 ML Problem

**Goal**: Accurately forecast the next hour’s electricity demand across NYC.

This enables:
- 🔄 Efficient load balancing
- 📉 Reduced over/under production
- 😊 Improved customer satisfaction

---

## 🔍 Data Sources

- 📊 **Electricity Demand Data** from [EIA API](https://www.eia.gov/opendata/)
- 🌦️ **Weather Data** from [Open-Meteo API](https://open-meteo.com/)
- 📅 **Calendar Data** from `pandas.tseries.holiday`

---

## 🔧 Project Methodology

The solution uses a **three-stage ML pipeline**, integrating feature stores, model registries, and automated inference via **Hopsworks**.

---

### 🔹 1. Feature Pipeline

📥 Ingests and processes raw data to generate features.

#### Steps:
1. Retrieve demand from **EIA API**
2. Get weather info from **Open-Meteo**
3. Merge data sources by timestamp
4. Transform to time-series format
5. Store in **Hopsworks Feature Store**

---

### 🔹 2. Training Pipeline

🧪 Trains a robust model and registers it for inference.

#### Steps:
1. Load time-series data from feature store
2. Create feature-target pairs:
   - Target: Hourly electricity demand
   - Features: Temp, lags, calendar holidays
3. Train **LightGBM** using **Optuna** with 5-fold CV
4. Add time-series & calendar-based features
5. Evaluate with **Mean Absolute Error (MAE)**
6. Register model in **Hopsworks Model Registry**

---

### 🔹 3. Inference Pipeline

📈 Generates live hourly forecasts.

#### Steps:
1. Fetch current features from Feature Store
2. Load model from Model Registry
3. Predict hourly demand
4. Compare vs actual, compute MAE
5. Runs **hourly via GitHub Actions**

---

## 🚀 Deployment & Monitoring

We built two Streamlit apps for interactivity and transparency.

---

### 📍 Batch Forecasting App

🗺️ An interactive Streamlit app shows:
- Predicted demand across NYC
- Circle size represents demand level
- Time-based demand visualization

---

### 📊 Monitoring Dashboard

📡 Real-time performance monitoring with:
- Live MAE trends
- Historical predictions vs actuals
- Error analysis dashboard

---

## ✅ Summary

🎯 This system enables **public energy companies** to:

- 🔁 Automate data → prediction pipeline
- 🧠 Leverage Feature Store & Model Registry
- 📍 Forecast hourly demand across locations
- 📈 Visualize results with an intuitive dashboard
- 🏭 Operate at scale with retraining and monitoring

---

## 👨‍💻 Tech Stack

- ⚙️ **MLOps**: Hopsworks, Feature Store, Model Registry  
- 💻 **ML**: LightGBM, Optuna, Pandas, Scikit-learn  
- ☁️ **Scheduling**: GitHub Actions  
- 🖥️ **App**: Streamlit  
- 📈 **Monitoring**: MAE visualization dashboard  

---
