# src/api/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional

class TextIn(BaseModel):
    text: str = Field(..., example="Scientists discover new exoplanet with potential for life.")

class TextsIn(BaseModel):
    texts: List[str] = Field(..., example=["Global markets rally on positive economic news.", "New AI breakthrough in natural language understanding."])

class PredictionOut(BaseModel):
    category: str
    confidence: float # Or probability

class PredictionsOut(BaseModel):
    predictions: List[Dict[str, Union[str, float]]] # List of {"category": "...", "probability": 0.X}

class ModelMetrics(BaseModel): # Create a sub-model for metrics
    accuracy: Optional[float] = None
    f1_score_weighted: Optional[float] = None
    val_loss: Optional[float] = None # Make val_loss specifically optional

class ModelInfoOut(BaseModel):
    model_name: str
    model_version: str
    model_stage: str # This will hold the alias
    model_description: Union[str, None] = None
    model_metrics: Optional[ModelMetrics] = None # Use the sub-model, also make it optional
    artifact_path: Union[str, None] = None
    # Add run_id if you want to expose it, but it's not in the original ModelInfoOut
    # run_id: Optional[str] = None
