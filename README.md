# MLOps Project: NLP Pipeline for Text Classification

This project implements an end-to-end MLOps pipeline for a text classification task, specifically categorizing news headlines. It covers data ingestion and preprocessing using Apache Airflow, model experimentation and tracking with MLflow, model serving via a FastAPI REST API, and monitoring with Prometheus and Grafana.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Directory Structure](#directory-structure)
5. [Setup and Installation](#setup-and-installation)
    * [Prerequisites](#prerequisites)
    * [Dataset and Embeddings](#dataset-and-embeddings)
    * [Environment Setup](#environment-setup)
    * [Configuration](#configuration)
6. [Running the Pipeline](#running-the-pipeline)
    * [1. Start MLflow Tracking Server](#1-start-mlflow-tracking-server)
    * [2. Start Apache Airflow (via Docker Compose)](#2-start-apache-airflow-via-docker-compose)
    * [3. Run the Airflow DAG](#3-run-the-airflow-dag)
    * [4. Run Model Training Scripts](#4-run-model-training-scripts)
    * [5. Start the FastAPI Model Serving API](#5-start-the-fastapi-model-serving-api)
    * [6. Start Prometheus](#6-start-prometheus)
    * [7. Start Grafana](#7-start-grafana)
7. [Usage](#usage)
    * [Accessing Airflow UI](#accessing-airflow-ui)
    * [Accessing MLflow UI](#accessing-mlflow-ui)
    * [Accessing FastAPI (Swagger UI & Demo)](#accessing-fastapi-swagger-ui--demo)
    * [Accessing Prometheus UI](#accessing-prometheus-ui)
    * [Accessing Grafana UI](#accessing-grafana-ui)
8. [Technologies Used](#technologies-used)
9. [Challenges and Learnings](#challenges-and-learnings)
10. [Future Work](#future-work)

## Project Overview

The goal of this project is to build a robust MLOps pipeline demonstrating best practices for developing, deploying, and monitoring a machine learning model. The task is to classify news headlines from the "News Category Dataset" into predefined categories (e.g., Business, Technology, Sports).

## Features

*   **Automated Data Processing:** Apache Airflow DAG for ingesting, cleaning, preprocessing, feature engineering, and splitting the dataset.
*   **Experiment Tracking:** MLflow for logging multiple model experiments, including parameters, metrics, and artifacts.
*   **Model Registry:** MLflow Model Registry for versioning and managing trained models.
*   **Model Serving:** FastAPI REST API to serve the best model for real-time predictions (single and batch).
*   **API Documentation:** Automatic Swagger UI and ReDoc generation by FastAPI.
*   **Monitoring:** Prometheus for collecting API and model metrics, and Grafana for visualizing these metrics on dashboards.
*   **Containerization:** Docker and Docker Compose used for running Airflow, Prometheus, and Grafana.

## Architecture

The project follows a standard MLOps workflow:

1.  **Data Pipeline (Apache Airflow):** Ingests raw data, preprocesses it (cleaning, lemmatization, feature engineering), and splits it into training, validation, and test sets. These processed datasets are versioned implicitly by Airflow run or stored for MLflow.
2.  **Model Training & Tracking (MLflow):** Various models (TF-IDF with Naive Bayes/Logistic Regression, Keras NNs with custom/pre-trained embeddings, Hugging Face Transformers) are trained. Experiments, parameters, metrics, and model artifacts are tracked using MLflow. The best model is registered in the MLflow Model Registry.
3.  **Model Serving (FastAPI):** A REST API loads a chosen model version (e.g., "Production" alias) from the MLflow Model Registry and serves predictions.
4.  **Monitoring (Prometheus & Grafana):** The FastAPI is instrumented to expose metrics. Prometheus scrapes these metrics, and Grafana provides dashboards for visualization of API performance, prediction distributions, and basic data drift indicators.

## Setup and Installation

### Prerequisites

*   Python (3.8 - 3.11 recommended)
*   pip (Python package installer)
*   Docker and Docker Compose
*   Git and Git LFS (Git LFS only if you choose to track large files, otherwise ignore them)
*   An MLflow Tracking Server (local or remote)
*   (Optional) A virtual environment (e.g., `venv`, `conda`) is highly recommended.

### Dataset and Embeddings

1.  **News Category Dataset:**
    *   Download from Kaggle: [https://www.kaggle.com/datasets/rmisra/news-category-dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (Version 3, `News_Category_Dataset_v3.json`).
    *   Place the downloaded `News_Category_Dataset_v3.json` file into the `mlops-nlp-project/data/raw/` directory.
2.  **GloVe Embeddings (Optional, if running GloVe experiment):**
    *   Download `glove.6B.100d.txt` (or your preferred dimension, e.g., 50d, 200d, 300d) from [Stanford GloVe page](https://nlp.stanford.edu/projects/glove/).
    *   Place the downloaded file (e.g., `glove.6B.100d.txt`) into the `mlops-nlp-project/embeddings/` directory.
    *   Ensure the `GLOVE_EMBEDDING_PATH` in `scripts/train.py` matches the filename and the `EMBEDDING_DIM_GLOVE` matches the dimension chosen.

### Environment Setup

1.  **Clone the repository (if you haven't):**
    ```bash
    git clone <your-repo-url>
    cd mlops-nlp-project
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
4.  **Download NLTK resources and spaCy model (if used by `nlp_utils.py`):**
    Run these in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4') # For WordNet lemmatization

    # If spaCy is used in nlp_utils.py for advanced_processing
    # import spacy
    # spacy.cli.download("en_core_web_sm")
    ```

### Configuration

*   **Airflow Variables (Optional but Recommended):** In the Airflow UI (Admin -> Variables), you can set:
    *   `random_seed`: e.g., `42`
    *   `test_size`: e.g., `0.2`
    *   `validation_size`: e.g., `0.1`
    *   `sample_fraction`: e.g., `0.1` (for DAG data sampling)
    The DAG (`news_processing_dag.py`) uses defaults if these variables are not found.
*   **API Configuration (`src/api/main.py`):**
    *   `MLFLOW_TRACKING_URI`: Defaults to `http://localhost:5000`. Set environment variable or change default if your MLflow server is elsewhere.
    *   `REGISTERED_MODEL_NAME`: Defaults to `NewsCategoryClassifier`.
    *   `MODEL_ALIAS`: Defaults to `Production`. This is the alias used to fetch the model from the registry.
*   **Prometheus (`prometheus.yml`):**
    *   Ensure the `targets` under `scrape_configs` for `fastapi_app` points to your running FastAPI app's `/metrics` endpoint (e.g., `host.docker.internal:8000` if FastAPI is on host and Prometheus in Docker Desktop).

## Running the Pipeline

Execute these steps in order, typically in separate terminal windows.

### 1. Start MLflow Tracking Server

Navigate to the project root (`mlops-nlp-project/`) and run:
```bash
# Ensure directories exist:
mkdir -p mlflow_tracking_server_data mlflow_artifacts

mlflow server --backend-store-uri ./mlflow_tracking_server_data \
              --default-artifact-root ./mlflow_artifacts \
              --host 0.0.0.0 --port 5000

Keep this terminal running.
2. Start Apache Airflow (via Docker Compose)
Navigate to mlops-nlp-project/airflow/dags/ and run:

# Create required directories for Airflow Docker Compose if they don't exist
# (relative to the location of docker-compose.yaml - which is airflow/dags/)
mkdir -p ../logs ../plugins ../config

# Create .env file for AIRFLOW_UID (in airflow/dags/)
echo "AIRFLOW_UID=$(id -u)" > .env # For Linux/macOS
# For Windows (Git Bash/MINGW64), use a fixed UID like 50000:
# echo "AIRFLOW_UID=50000" > .env

# Initialize Airflow database (first time only)
docker-compose up airflow-init

# Start all Airflow services
docker-compose up -d # or 'docker-compose up' to see logs

Keep these services running.
3. Run the Airflow DAG
Access Airflow UI: http://localhost:8080 (default login: airflow/airflow).
Find the news_category_data_processing_v1 DAG.
Unpause it (toggle on the left).
Trigger it manually (play button) for the initial data processing run.
It is scheduled to run daily at 2:00 AM UTC by default.
4. Run Model Training Scripts
Ensure your Python virtual environment is active.
Navigate to the project root (mlops-nlp-project/).
Ensure the processed data from Airflow exists in data/processed/.
Run the training script:

python scripts/train.py

Experiments will be logged to the MLflow server started in Step 1.
After training, go to the MLflow UI (http://localhost:5000), select your best model run, and register it as NewsCategoryClassifier (or your configured name). Then, assign it an alias (e.g., Production) via the "Aliases: Add" link on the model version page.
5. Start the FastAPI Model Serving API
Ensure your Python virtual environment is active.
Ensure the MLflow Tracking Server is running (from Step 1).
Ensure a model version has the alias specified by MODEL_ALIAS (e.g., "Production") in the MLflow Model Registry.
Navigate to the project root (mlops-nlp-project/).
Run Uvicorn:

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Keep this terminal running.
6. Start Prometheus
Ensure Docker is running.
Navigate to the project root (mlops-nlp-project/).
Ensure prometheus.yml is configured correctly.
Run Prometheus Docker container:

# For Docker Desktop (Windows/Mac)
docker run -d --name prometheus_monitoring -p 9090:9090 -v "${PWD}/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus:latest
# For Linux, if using host network for Prometheus:
# docker run -d --name prometheus_monitoring --network="host" -v "$(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus:latest
# (and ensure prometheus.yml target is localhost:8000)

Start Grafana
Ensure Docker is running.
Navigate to the project root (mlops-nlp-project/).
Run Grafana Docker container:

docker run -d --name grafana_monitoring -p 3000:3000 grafana/grafana-oss:latest

Access Grafana UI (http://localhost:3000), log in (admin/admin), change password.
Add Prometheus as a data source:
URL: http://host.docker.internal:9090 (if Prometheus is Dockerized on Docker Desktop and API is on host) or http://<prometheus_container_ip_or_name>:9090 or http://localhost:9090 (if Prometheus uses host network).
Create dashboards as described in the project guides.
Usage
Accessing Airflow UI
URL: http://localhost:8080
Credentials: airflow / airflow (default from official Airflow Docker Compose)
Accessing MLflow UI
URL: http://localhost:5000 (or your configured MLflow server address)
Accessing FastAPI (Swagger UI & Demo)
Swagger UI (API Docs): http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
Simple HTML Demo: Open api_demo.html from the project root in your browser (file:///.../api_demo.html). Ensure FastAPI CORS is configured if you see errors.
Accessing Prometheus UI
URL: http://localhost:9090
Check Status -> Targets to ensure fastapi_app is UP.
Accessing Grafana UI
URL: http://localhost:3000
Credentials: admin / admin (change on first login).
View your created dashboards.
Technologies Used
Python: Core programming language.
Apache Airflow: Workflow orchestration for data pipelines.
MLflow: Experiment tracking, model registry, and model packaging.
FastAPI: High-performance web framework for building the REST API.
Uvicorn: ASGI server for FastAPI.
Pydantic: Data validation for API models.
Prometheus: Metrics collection and time-series database.
Grafana: Metrics visualization and dashboarding.
Docker & Docker Compose: Containerization and multi-container application management.
Scikit-learn: For traditional ML models and preprocessing.
TensorFlow/Keras: For neural network models.
Hugging Face Transformers: For transformer-based models (DistilBERT).
NLTK, spaCy: NLP libraries for text preprocessing.
Pandas, NumPy: Data manipulation.
Git & GitHub: Version control and code hosting.
Challenges and Learnings
Setting up Docker networking between containers (e.g., Grafana to Prometheus, Prometheus to FastAPI).
Managing Python dependencies across different components.
Debugging MLflow model loading in FastAPI, especially with custom preprocessors.
Understanding PromQL for creating effective Grafana dashboards.
Git history management with large files.
Learning the nuances of each MLOps tool and how they integrate.)
