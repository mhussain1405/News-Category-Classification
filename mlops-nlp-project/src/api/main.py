# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.logger import logger as fastapi_logger # For structured logging
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
import uvicorn
import os
import sys
import joblib
import numpy as np
import pandas as pd
import mlflow
from typing import List, Dict, Any, Union, Optional # Added Optional
import logging # Standard Python logging

# Prometheus instrumentation imports
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter as PrometheusCounter, Histogram as PrometheusHistogram

# Add project root to sys.path to import nlp_utils and other modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.nlp_utils import basic_cleaning, advanced_processing
from src.api.models import TextIn, TextsIn, PredictionOut, PredictionsOut, ModelInfoOut, ModelMetrics # Added ModelMetrics

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "NewsCategoryClassifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")

MODEL_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'models')

# --- Logging Configuration ---
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Global Variables for Model and Preprocessors ---
ml_model = None
tfidf_vectorizer_global = None
keras_tokenizer_global = None
hf_tokenizer_global = None
label_encoder_global = None
model_info_global = {}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="News Category Classification API",
    description="API for classifying news headlines into categories.",
    version="1.1.0" # Incremented version for monitoring
)

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:8080", # If your demo or other frontend runs here
    "null",                  # For 'file://' origins (local HTML file)
    "*"                      # Allows all origins - Be cautious in production
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prometheus Metrics Configuration ---
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/docs", "/openapi.json"], # Exclude metrics and docs
    inprogress_labels=True,
)
# Instrument the app after all routes are defined if you want to ensure all are caught
# Or instrument early. For custom metrics, it doesn't matter as much when this is called.
# Expose /metrics endpoint
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)


# Custom Prometheus Metrics
PREDICTION_CATEGORY_COUNTER = PrometheusCounter(
    "api_prediction_category_total",
    "Counts predictions by category and model flavor.",
    ["model_flavor", "predicted_category"]
)
INPUT_TEXT_LENGTH_HISTOGRAM = PrometheusHistogram(
    "api_input_text_length_chars",
    "Histogram of input text lengths in characters.",
    ["model_flavor"],
    buckets=(10, 25, 50, 75, 100, 150, 200, 300, 500, float("inf")) # Adjusted buckets
)
CUSTOM_API_ERROR_COUNTER = PrometheusCounter(
    "api_custom_error_total",
    "Counts specific application errors.",
    ["endpoint", "error_type"]
)

# --- Model Loading Logic ---
def load_model_from_mlflow(model_name: str, alias: str):
    global ml_model, tfidf_vectorizer_global, keras_tokenizer_global, hf_tokenizer_global, label_encoder_global, model_info_global
    
    logger.info(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        client = mlflow.tracking.MlflowClient()
        logger.info(f"Attempting to load model '{model_name}' with alias '{alias}'")

        try:
            model_version_details = client.get_model_version_by_alias(name=model_name, alias=alias)
            model_version = model_version_details.version
            run_id = model_version_details.run_id
            run_artifact_uri = client.get_run(run_id).info.artifact_uri

            model_info_global = {
                "model_name": model_name,
                "model_version": model_version,
                "model_stage": alias, 
                "model_description": model_version_details.description,
                "artifact_path": run_artifact_uri,
                "run_id": run_id
            }
            logger.info(f"Found model: {model_name} version {model_version} with alias '{alias}' from run {run_id}")

            try:
                run_data = client.get_run(run_id).data
                metrics_data = {
                    "accuracy": run_data.metrics.get("accuracy"),
                    "f1_score_weighted": run_data.metrics.get("f1_score_weighted"),
                    "val_loss": run_data.metrics.get("val_loss")
                }
                # Use the ModelMetrics Pydantic model for validation and structure
                model_info_global["model_metrics"] = ModelMetrics(**{k: v for k, v in metrics_data.items() if v is not None})
            except Exception as e:
                logger.warning(f"Could not retrieve or parse metrics for run {run_id}: {e}")
                model_info_global["model_metrics"] = None

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"MLflow Registry error: Could not get model version for alias '{alias}': {e}")
            raise HTTPException(status_code=503, detail=f"Model '{model_name}@{alias}' not found in registry or registry unavailable.")
        except Exception as e:
            logger.error(f"Unexpected error fetching model version details: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error fetching model version details.")

        try:
            le_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="label_encoder_artifact/label_encoder.joblib")
            label_encoder_global = joblib.load(le_local_path)
            logger.info("Label encoder loaded successfully from run artifacts.")
        except Exception as e:
            logger.error(f"Failed to load label_encoder from run artifacts: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Critical error: Label encoder could not be loaded from run artifacts.")
            
        artifact_list = [f.path for f in client.list_artifacts(run_id)]
        logger.debug(f"Artifacts in run {run_id}: {artifact_list}")

        model_loaded_successfully = False
        if any("naive_bayes_model" in path or "svm_model" in path for path in artifact_list):
            logger.info("Detected scikit-learn model type.")
            model_path_in_run = next((p for p in ["naive_bayes_model", "svm_model"] if any(p in art_path for art_path in artifact_list)), None)
            
            if model_path_in_run:
                ml_model = mlflow.sklearn.load_model(model_uri=f"runs:/{run_id}/{model_path_in_run}")
                tfidf_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="tfidf_vectorizer_artifact/tfidf_vectorizer.joblib")
                tfidf_vectorizer_global = joblib.load(tfidf_local_path)
                logger.info(f"Loaded scikit-learn model '{model_path_in_run}' and TF-IDF vectorizer from run {run_id}.")
                model_info_global["flavor"] = "sklearn"
                model_loaded_successfully = True
            else:
                logger.error("Could not determine specific sklearn model artifact path within the run.")

        elif any("simple_nn_keras_model" in path or "glove_nn_keras_model" in path for path in artifact_list):
            logger.info("Detected Keras model type.")
            model_path_in_run = next((p for p in ["simple_nn_keras_model", "glove_nn_keras_model"] if any(p in art_path for art_path in artifact_list)), None)

            if model_path_in_run:
                ml_model = mlflow.keras.load_model(model_uri=f"runs:/{run_id}/{model_path_in_run}", custom_objects=None)
                keras_tokenizer_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="keras_tokenizer_artifact/keras_tokenizer.joblib")
                keras_tokenizer_global = joblib.load(keras_tokenizer_local_path)
                logger.info(f"Loaded Keras model '{model_path_in_run}' and Keras tokenizer from run {run_id}.")
                model_info_global["flavor"] = "keras"
                run_params = client.get_run(run_id).data.params
                model_info_global["keras_max_length"] = int(run_params.get("nn_max_seq_length", 100)) 
                model_loaded_successfully = True
            else:
                logger.error("Could not determine specific Keras model artifact path within the run.")

        elif any(a.startswith("distilbert_tf_model") for a in artifact_list):
            logger.info("Detected Hugging Face Transformer model type (DistilBERT).")
            model_dir_artifact_path_in_run = "distilbert_tf_model"
            
            local_model_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=model_dir_artifact_path_in_run)
            
            from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
            ml_model = TFAutoModelForSequenceClassification.from_pretrained(local_model_dir)
            hf_tokenizer_global = AutoTokenizer.from_pretrained(local_model_dir)

            logger.info(f"Loaded DistilBERT model and tokenizer from run {run_id}, downloaded to {local_model_dir}.")
            model_info_global["flavor"] = "transformers"
            run_params = client.get_run(run_id).data.params
            model_info_global["hf_max_length"] = int(run_params.get("hf_max_length", 128))
            model_loaded_successfully = True
        
        if not model_loaded_successfully:
            logger.error("Model could not be loaded based on inferred type or missing artifacts.")
            raise HTTPException(status_code=500, detail="Unsupported model type or model artifacts not found as expected in the run.")
        
        logger.info(f"Successfully loaded model flavor '{model_info_global.get('flavor')}' (Version: {model_version}, Alias: '{alias}') from Run ID: {run_id}.")

    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow exception during model loading: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Error connecting to MLflow or loading model: {str(e)}")
    except Exception as e:
        logger.error(f"General exception during model loading: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during model loading: {str(e)}")

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Loading model...")
    load_model_from_mlflow(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    if ml_model is None:
        logger.error("Model could not be loaded at startup. API may not function correctly.")
    else:
        logger.info("Model loaded successfully. API is ready.")

# --- API Endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    if ml_model is not None and label_encoder_global is not None:
        return {"status": "healthy", "message": "API is up and model is loaded."}
    return {"status": "unhealthy", "message": "API is up but model is NOT loaded."}

@app.get("/model_info", response_model=ModelInfoOut, tags=["Model Information"])
async def get_model_info():
    if not model_info_global or ml_model is None:
        raise HTTPException(status_code=404, detail="Model information not available or model not loaded.")
    return ModelInfoOut(**model_info_global)

def preprocess_text_for_model(texts: List[str], flavor: str) -> Any:
    logger.debug(f"Preprocessing {len(texts)} texts for flavor: {flavor}")
    cleaned_texts = [basic_cleaning(text) for text in texts]
    processed_texts = [advanced_processing(text) for text in cleaned_texts]

    if flavor == "sklearn":
        if tfidf_vectorizer_global is None: raise RuntimeError("TF-IDF vectorizer not loaded.")
        return tfidf_vectorizer_global.transform(processed_texts)
    elif flavor == "keras":
        if keras_tokenizer_global is None: raise RuntimeError("Keras tokenizer not loaded.")
        max_length = model_info_global.get("keras_max_length", 100)
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        sequences = keras_tokenizer_global.texts_to_sequences(processed_texts)
        return pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    elif flavor == "transformers":
        if hf_tokenizer_global is None: raise RuntimeError("Hugging Face tokenizer not loaded.")
        max_length_hf = model_info_global.get("hf_max_length", 128)
        return hf_tokenizer_global(processed_texts, truncation=True, padding=True, max_length=max_length_hf, return_tensors="tf", add_special_tokens=True)
    else:
        raise ValueError(f"Unsupported model flavor for preprocessing: {flavor}")

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
async def predict_single(item: TextIn):
    if ml_model is None or label_encoder_global is None:
        logger.error("Prediction: Model or label encoder not loaded.")
        CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict", error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        flavor = model_info_global.get("flavor")
        if not flavor:
            CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict", error_type="flavor_not_determined").inc()
            raise RuntimeError("Model flavor not determined for prediction.")
        
        logger.info(f"Single predict request for text: '{item.text[:50]}...' with flavor: {flavor}")
        INPUT_TEXT_LENGTH_HISTOGRAM.labels(model_flavor=flavor).observe(len(item.text))
        processed_input = preprocess_text_for_model([item.text], flavor)

        if flavor == "sklearn":
            prediction_probs = ml_model.predict_proba(processed_input)[0]
        elif flavor == "keras":
            prediction_probs = ml_model.predict(processed_input)[0]
        elif flavor == "transformers":
            outputs = ml_model(processed_input)
            logits = outputs.logits[0]
            from tensorflow.nn import softmax
            prediction_probs = softmax(logits).numpy()
        else:
            CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict", error_type="unsupported_flavor").inc()
            raise ValueError(f"Unsupported model flavor for prediction: {flavor}")

        top_category_idx = np.argmax(prediction_probs)
        top_category_label = label_encoder_global.classes_[top_category_idx]
        top_confidence = float(prediction_probs[top_category_idx])
        PREDICTION_CATEGORY_COUNTER.labels(model_flavor=flavor, predicted_category=top_category_label).inc()
        
        logger.info(f"Prediction: {top_category_label} (Confidence: {top_confidence:.4f})")
        return PredictionOut(category=top_category_label, confidence=top_confidence)
    except RuntimeError as e:
        logger.error(f"Runtime error during prediction: {e}", exc_info=True)
        CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict", error_type="runtime_error").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during single prediction: {e}", exc_info=True)
        CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict", error_type="unexpected_error").inc()
        raise HTTPException(status_code=500, detail="Error processing request.")

@app.post("/predict_batch", response_model=PredictionsOut, tags=["Prediction"])
async def predict_batch(items: TextsIn):
    if ml_model is None or label_encoder_global is None:
        logger.error("Batch predict: Model or label encoder not loaded.")
        CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict_batch", error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        flavor = model_info_global.get("flavor")
        if not flavor:
            CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict_batch", error_type="flavor_not_determined").inc()
            raise RuntimeError("Model flavor not determined for batch prediction.")
        
        logger.info(f"Batch predict request for {len(items.texts)} texts with flavor: {flavor}.")
        for text_item in items.texts:
            INPUT_TEXT_LENGTH_HISTOGRAM.labels(model_flavor=flavor).observe(len(text_item))
        processed_inputs = preprocess_text_for_model(items.texts, flavor)
        all_predictions = []

        if flavor == "sklearn":
            batch_prediction_probs = ml_model.predict_proba(processed_inputs)
        elif flavor == "keras":
            batch_prediction_probs = ml_model.predict(processed_inputs)
        elif flavor == "transformers":
            outputs = ml_model(processed_inputs)
            logits_batch = outputs.logits
            from tensorflow.nn import softmax
            batch_prediction_probs = softmax(logits_batch).numpy()
        else:
            CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict_batch", error_type="unsupported_flavor").inc()
            raise ValueError(f"Unsupported model flavor for batch prediction: {flavor}")

        for i, text_probs in enumerate(batch_prediction_probs):
            top_category_idx = np.argmax(text_probs)
            top_category_label = label_encoder_global.classes_[top_category_idx]
            top_confidence = float(text_probs[top_category_idx])
            all_predictions.append({"category": top_category_label, "probability": top_confidence})
            PREDICTION_CATEGORY_COUNTER.labels(model_flavor=flavor, predicted_category=top_category_label).inc()
            if i < 3: logger.debug(f"Batch pred for '{items.texts[i][:50]}...': {top_category_label} ({top_confidence:.4f})")
        
        logger.info(f"Processed batch of {len(items.texts)} texts.")
        return PredictionsOut(predictions=all_predictions)
    except RuntimeError as e:
        logger.error(f"Runtime error during batch prediction: {e}", exc_info=True)
        CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict_batch", error_type="runtime_error").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during batch prediction: {e}", exc_info=True)
        CUSTOM_API_ERROR_COUNTER.labels(endpoint="/predict_batch", error_type="unexpected_error").inc()
        raise HTTPException(status_code=500, detail="Error processing batch request.")

# --- Main block to run Uvicorn ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly for development...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level=log_level_str.lower(), reload=True)
    # Note: If running this file directly, and it's in src/api/main.py,
    # uvicorn might need the path relative to where you run python.
    # So, if you run `python src/api/main.py` from project root:
    # uvicorn.run(app, ...) is fine.
    # If you rely on uvicorn CLI: `uvicorn src.api.main:app --reload` from project root.
    # The "main:app" in uvicorn.run() assumes the file is named main.py and 'app' is the FastAPI instance.