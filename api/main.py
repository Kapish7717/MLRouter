# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time, hashlib, json

from serving.model_registry import registry
from serving.router         import router, ExperimentConfig
from tracking.database      import (log_prediction,
                                    get_experiment_metrics,
                                    get_recent_predictions)

app = FastAPI(
    title="ML Model Serving Platform",
    description="A/B testing platform for ML models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features:       Dict[str, Any]
    experiment_id:  str = "exp_001"

class PredictResponse(BaseModel):
    prediction:     int
    confidence:     float
    model_id:       str
    variant:        str         # A or B
    latency_ms:     float
    experiment_id:  str

class ExperimentUpdateRequest(BaseModel):
    traffic_split:  float       # 0.0 to 1.0

# ── Core prediction endpoint ──────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Main prediction endpoint.
    1. Routes to model A or B based on A/B config
    2. Runs inference
    3. Logs everything to database
    4. Returns prediction with metadata
    """

    # Step 1: Route to correct model
    model_id, variant = router.route(request.experiment_id)

    # Step 2: Get model from registry
    registered = registry.get(model_id)

    # Step 3: Prepare features
    try:
        f = request.features
        
        # map categorical fields that load_test sends as raw
        vitamin_d = f.get("Vitamin D Intake", 0)
        age = f.get("Age", 50)
        scaled_age = (age - 55.0) / 15.0  # approximate scaling
        
        row = {
            'Age': scaled_age,
            'Prior Fractures': f.get("Prior Fractures", 0),
            'Family History': f.get("Family History", 0),
            'Gender': f.get("Gender", 0),
            'Vitamin D Intake_Sufficient': vitamin_d,
            'Physical Activity': f.get("Physical Activity", 0)
        }
        import pandas as pd
        df = pd.DataFrame([row])
        
        # Make sure the dataframe exactly matches what the model expects
        model_obj = registered.model
        if hasattr(model_obj, "feature_names_in_"):
            expected = list(model_obj.feature_names_in_)
            if any("_" in f for f in expected) and not any(" " in f for f in expected):
                # LightGBM replaced spaces with underscores
                df.columns = [c.replace(' ', '_') for c in df.columns]
            
            # order columns as the model expects
            df = df[expected]
            
    except KeyError as e:
        raise HTTPException(
            400, f"Missing feature: {e}"
        )

    # Step 4: Run inference + measure latency
    start  = time.time()
    pred   = int(model_obj.predict(df)[0])
    
    # get proba of predicted class or class 1
    proba = model_obj.predict_proba(df)[0]
    conf = float(proba[1] if len(proba) > 1 else proba[0])
    lat_ms = round((time.time() - start) * 1000, 2)

    # Step 5: Log to database
    input_hash = hashlib.md5(
        json.dumps(request.features, sort_keys=True
    ).encode()).hexdigest()[:8]

    log_prediction(
        experiment_id= request.experiment_id,
        model_id=      model_id,
        variant=       variant,
        prediction=    pred,
        confidence=    conf,
        latency_ms=    lat_ms,
        input_hash=    input_hash
    )

    return PredictResponse(
        prediction=    pred,
        confidence=    round(conf, 4),
        model_id=      model_id,
        variant=       variant,
        latency_ms=    lat_ms,
        experiment_id= request.experiment_id
    )

# ── Experiment management endpoints ──────────────────────────────
@app.get("/experiments/{experiment_id}/metrics")
async def get_metrics(experiment_id: str):
    """Get live A/B comparison metrics"""
    return {
        "experiment_id": experiment_id,
        "metrics":       get_experiment_metrics(experiment_id)
    }

@app.get("/experiments/{experiment_id}/predictions")
async def get_predictions(
    experiment_id: str, limit: int = 50
):
    """Get recent predictions for analysis"""
    return get_recent_predictions(experiment_id, limit)

@app.patch("/experiments/{experiment_id}/split")
async def update_split(
    experiment_id: str,
    body: ExperimentUpdateRequest
):
    """
    Update traffic split on the fly.
    Start 50/50, move to 80/20 as you gain confidence.
    """
    router.update_split(
        experiment_id, body.traffic_split
    )
    return {
        "message": "Split updated",
        "new_split": body.traffic_split
    }

@app.post("/experiments/{experiment_id}/promote")
async def promote_winner(
    experiment_id: str, winner: str
):
    """
    End experiment and route 100% to winner.
    winner = 'A' or 'B'
    """
    router.stop_experiment(experiment_id, winner)
    return {
        "message":        f"Model {winner} promoted to 100%",
        "experiment_id":  experiment_id
    }

@app.get("/models")
async def list_models():
    """List all registered models"""
    return {"models": registry.list_models()}

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": len(registry.models)}