# Real-Time Banking Transaction Fraud Detection Using Big Data

## 📌 Project Overview

This project is a real-time banking transaction fraud detection system that uses Big Data technologies and Machine Learning to analyze transactions and identify potentially fraudulent activity.

The system processes transactions through Apache Kafka and Apache Spark Streaming, applies a trained Random Forest machine learning model, and returns a fraud probability and classification through a FastAPI backend. A web-based frontend allows users to submit transactions and view the prediction in real time.

## 🎯 Objectives

- Real-time transaction processing
- Machine learning-based fraud prediction
- Distributed stream processing using Apache Spark
- Transaction streaming using Apache Kafka
- Instant fraud detection results
- Scalable architecture for transaction analysis
- Web-based interface for transaction analysis

## 🏗️ System Architecture

```text
User
  ↓
Frontend
  ↓
FastAPI
  ↓
Apache Kafka
  ↓
Spark Streaming
  ↓
Random Forest ML Model
  ↓
Fraud Probability
  ↓
Kafka
  ↓
FastAPI
  ↓
Frontend Result
