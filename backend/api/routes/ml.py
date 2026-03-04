"""
ml.py — DiagnoSys API Routes
POST /ml/classify — proxy to ML inference service on port 8001.
"""

import os
import httpx
from fastapi import APIRouter, HTTPException
from schemas import ClassifyRequest, ClassifyResponse

router = APIRouter(prefix="/ml", tags=["ml"])
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml_classifier:8001")


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{ML_SERVICE_URL}/ml/classify",
                json={"text": request.text, "context": request.context},
            )
            resp.raise_for_status()
            return ClassifyResponse(**resp.json())
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="ML service timeout")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"ML service error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
