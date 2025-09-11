from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import os
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Placement Prediction API",
    description="Predict student placement based on CGPA and IQ using Machine Learning",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    logger.info("Model and scaler loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Model files not found: {e}")
    model = None
    scaler = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    scaler = None

# Request/Response models
class PredictionRequest(BaseModel):
    cgpa: float
    iq: float
    
class PredictionResponse(BaseModel):
    cgpa: float
    iq: float
    placement_prediction: int
    probability: float
    confidence: str

class BatchPredictionRequest(BaseModel):
    students: List[PredictionRequest]

@app.get("/")
async def root():
    return {
        "message": "🎓 Placement Prediction ML API is running!",
        "status": "healthy",
        "model_loaded": model is not None,
        "endpoints": ["/predict", "/predict_batch", "/health", "/docs"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_placement(request: PredictionRequest):
    """
    Predict placement probability for a single student
    
    - **cgpa**: Student's CGPA (0.0 to 10.0)
    - **iq**: Student's IQ score (typically 80-150)
    
    Returns placement prediction (0 = Not Placed, 1 = Placed) with probability
    """
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Validate input ranges
    if not (0.0 <= request.cgpa <= 10.0):
        raise HTTPException(status_code=400, detail="CGPA must be between 0.0 and 10.0")
    
    if not (50 <= request.iq <= 200):
        raise HTTPException(status_code=400, detail="IQ must be between 50 and 200")
    
    try:
        # Prepare input data
        X = np.array([[request.cgpa, request.iq]])
        
        # Scale the input (assuming you saved the scaler)
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0].max()
        
        # Determine confidence level
        if probability >= 0.8:
            confidence = "High"
        elif probability >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return PredictionResponse(
            cgpa=request.cgpa,
            iq=request.iq,
            placement_prediction=int(prediction),
            probability=round(probability, 3),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict placement for multiple students at once
    """
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if len(request.students) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 students per batch")
    
    try:
        results = []
        
        for student in request.students:
            # Prepare input
            X = np.array([[student.cgpa, student.iq]])
            
            # Scale if scaler available
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0].max()
            
            # Confidence level
            if probability >= 0.8:
                confidence = "High"
            elif probability >= 0.6:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            results.append({
                "cgpa": student.cgpa,
                "iq": student.iq,
                "placement_prediction": int(prediction),
                "probability": round(probability, 3),
                "confidence": confidence
            })
        
        return {"predictions": results, "total_students": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": str(type(model).__name__),
        "features": ["cgpa", "iq"],
        "output": "placement (0=Not Placed, 1=Placed)",
        "model_loaded": True,
        "scaler_loaded": scaler is not None
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": ["/", "/predict", "/predict_batch", "/health", "/docs"]}

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please try again later"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)