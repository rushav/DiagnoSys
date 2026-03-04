"""
inference.py — DiagnoSys ML Classification
FastAPI inference service on port 8001.
POST /ml/classify  →  { domains, confidence, model_version }
Model is loaded once at startup and kept resident in GPU memory.
Target: <200 ms P50 latency.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from torch.cuda.amp import autocast

from dataset import clean_html, IDX2LABEL, NUM_LABELS, MAX_TOKEN_LEN

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = os.environ.get("BASE_MODEL",    "meta-llama/Meta-Llama-3.1-8B")
ADAPTER_PATH     = os.environ.get("ADAPTER_PATH",  "ml/classification/checkpoints/best")
MODEL_VERSION    = os.environ.get("MODEL_VERSION", "llama3.1-8b-qlora-v1")
THRESHOLD        = float(os.environ.get("CLASSIFY_THRESHOLD", "0.40"))
MAX_BATCH        = int(os.environ.get("MAX_BATCH_SIZE", "32"))

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Global model state (loaded once at startup) ───────────────────────────────
_model     = None
_tokenizer = None


def load_model():
    global _model, _tokenizer

    logger.info("Loading tokeniser from %s …", MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    adapter_path = Path(ADAPTER_PATH)
    if adapter_path.exists():
        logger.info("Loading base model in 4-bit …")
        peft_cfg = PeftConfig.from_pretrained(str(adapter_path))
        base = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path,
            num_labels=NUM_LABELS,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            problem_type="multi_label_classification",
        )
        base.config.pad_token_id = _tokenizer.pad_token_id
        logger.info("Attaching LoRA adapters from %s …", adapter_path)
        _model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        # Fallback: load base model without adapters (for cold-start / testing)
        logger.warning(
            "Adapter path %s not found — loading bare base model (untrained).", adapter_path
        )
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            problem_type="multi_label_classification",
        )
        _model.config.pad_token_id = _tokenizer.pad_token_id

    _model.eval()
    logger.info("Model ready ✓  (version=%s)", MODEL_VERSION)


# ── FastAPI lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # Cleanup on shutdown
    global _model, _tokenizer
    del _model, _tokenizer
    torch.cuda.empty_cache()


app = FastAPI(
    title="DiagnoSys Classification API",
    version="1.0.0",
    description="Multi-label domain classification for engineering problems.",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    text:    str            = Field(..., min_length=1, max_length=8192,
                                   description="Problem title + body text")
    context: Optional[str] = Field(None, max_length=2048,
                                   description="Optional surrounding context")


class ClassifyResponse(BaseModel):
    domains:       list[str]
    confidence:    float
    model_version: str


class BatchClassifyRequest(BaseModel):
    items: list[ClassifyRequest] = Field(..., max_items=MAX_BATCH)


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


# ── Core inference logic ──────────────────────────────────────────────────────
def _prepare_texts(requests: list[ClassifyRequest]) -> list[str]:
    texts = []
    for req in requests:
        body = clean_html(req.text)
        if req.context:
            body = clean_html(req.context) + " " + body
        texts.append(body)
    return texts


@torch.inference_mode()
def _run_inference(texts: list[str], threshold: float = THRESHOLD) -> list[dict]:
    encoding = _tokenizer(
        texts,
        max_length=MAX_TOKEN_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(_model.device)
    attention_mask = encoding["attention_mask"].to(_model.device)

    with autocast(dtype=torch.bfloat16):
        logits = _model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs = torch.sigmoid(logits).float().cpu().numpy()  # (B, NUM_LABELS)

    results = []
    for prob_row in probs:
        active_mask    = prob_row >= threshold
        active_domains = [IDX2LABEL[i] for i, flag in enumerate(active_mask) if flag]
        # Fallback: if nothing crosses threshold, pick argmax
        if not active_domains:
            best_idx       = int(np.argmax(prob_row))
            active_domains = [IDX2LABEL[best_idx]]
        confidence = float(np.mean(prob_row[active_mask]) if active_mask.any() else np.max(prob_row))
        results.append({
            "domains":       active_domains,
            "confidence":    round(confidence, 4),
            "model_version": MODEL_VERSION,
        })
    return results


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/ml/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify(request: ClassifyRequest):
    """
    Classify a single engineering problem into one or more domains.

    - **text**: raw problem text (HTML allowed — will be stripped)
    - **context**: optional surrounding context (e.g. repository description)

    Returns predicted **domains**, mean **confidence**, and **model_version**.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        results = _run_inference([request], threshold=THRESHOLD)
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("classify latency=%.1f ms  domains=%s", latency_ms, results[0]["domains"])

    return ClassifyResponse(**results[0])


@app.post("/ml/classify/batch", response_model=BatchClassifyResponse, tags=["Classification"])
async def classify_batch(request: BatchClassifyRequest):
    """
    Classify a batch of up to 32 engineering problems in a single forward pass.
    Significantly more efficient than calling /ml/classify in a loop.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(request.items) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Batch size exceeds limit of {MAX_BATCH}")

    t0 = time.perf_counter()
    try:
        results = _run_inference(request.items, threshold=THRESHOLD)
    except Exception as exc:
        logger.exception("Batch inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("classify_batch n=%d latency=%.1f ms", len(request.items), latency_ms)

    return BatchClassifyResponse(results=[ClassifyResponse(**r) for r in results])


@app.get("/health", tags=["Ops"])
async def health():
    """Liveness probe."""
    return {
        "status":        "ok",
        "model_loaded":  _model is not None,
        "model_version": MODEL_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "gpu":           torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/metrics", tags=["Ops"])
async def metrics():
    """Basic GPU memory metrics for monitoring."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved  = torch.cuda.memory_reserved(0)  / 1024 ** 3
        return {"gpu_allocated_gb": round(allocated, 2), "gpu_reserved_gb": round(reserved, 2)}
    return {"gpu_allocated_gb": 0, "gpu_reserved_gb": 0}


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    uvicorn.run(
        "inference:app",
        host="0.0.0.0",
        port=8001,
        workers=1,           # single worker — model lives in GPU memory
        log_level="info",
    )
