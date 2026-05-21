from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Request schemas ───────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """
    Clinical and epidemiological features for a single patient.

    All fields are optional — any missing value is imputed by the
    preprocessing pipeline (median for numeric, most-frequent for categorical).
    Do NOT send a 'label' field; it is the target variable.
    """
    model_config = ConfigDict(extra="forbid")

    age: float | None = Field(None, ge=0, le=120, description="Age in years")
    gender: str | None = Field(None, description="Gender: 'M' or 'F'")
    region: str | None = Field(
        None, description="Geographic region: north | south | east | west | central"
    )
    season: str | None = Field(
        None, description="Season: spring | summer | fall | winter"
    )
    rodent_contact: int | None = Field(None, ge=0, le=1, description="Rodent contact (0/1)")
    outdoor_work: int | None = Field(None, ge=0, le=1, description="Outdoor work (0/1)")
    fever: int | None = Field(None, ge=0, le=1, description="Fever present (0/1)")
    myalgia: int | None = Field(None, ge=0, le=1, description="Muscle pain (0/1)")
    headache: int | None = Field(None, ge=0, le=1, description="Headache (0/1)")
    thrombocytopenia: int | None = Field(
        None, ge=0, le=1, description="Low platelet count — strong HPS marker (0/1)"
    )
    rodent_density: float | None = Field(None, ge=0, description="Rodent density index (0–10)")
    precipitation_mm: float | None = Field(None, ge=0, description="Precipitation in mm")
    humidity_pct: float | None = Field(None, ge=0, le=100, description="Humidity percentage")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient: PatientInput
    model_name: str | None = Field(
        None,
        description="Optional: name of the model to use. Defaults to the best available.",
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patients: list[PatientInput] = Field(
        ..., min_length=1, max_length=100, description="1–100 patient records"
    )
    model_name: str | None = Field(
        None,
        description="Optional: name of the model to use. Defaults to the best available.",
    )


# ── Response schemas ──────────────────────────────────────────────────────────

class SinglePredictionResult(BaseModel):
    prediction: int = Field(description="0 = negative, 1 = positive")
    label: str = Field(description="'negative' or 'positive'")
    probability: float | None = Field(
        description="Calibrated probability after clinical stabilization (0.0–1.0)"
    )
    ml_raw_probability: float | None = Field(
        None,
        description="ML calibrated probability before clinical stabilization — for audit/explainability",
    )
    confidence: str = Field(description="'high' (≥0.85) | 'medium' (≥0.60) | 'low' | 'unknown'")
    near_threshold: bool = Field(
        False, description="True when probability is within 0.05 of a confidence threshold"
    )
    confidence_margin: float | None = Field(
        None, description="Distance to the nearest confidence threshold (0.85 or 0.60)"
    )
    stability_delta: float = Field(
        0.0, description="Net probability adjustment applied by clinical stabilizer"
    )
    stability_applied: bool = Field(
        False, description="Whether the clinical stabilizer produced a non-zero adjustment"
    )
    stability_contributions: list[dict] = Field(
        default_factory=list,
        description="Per-feature breakdown of stabilizer contributions",
    )
    model_name: str
    model_version: str
    inference_duration_ms: float
    timestamp: str


class BatchItemResult(BaseModel):
    prediction: int
    label: str
    probability: float | None
    confidence: str


class BatchPredictionResult(BaseModel):
    predictions: list[BatchItemResult]
    total: int
    model_name: str
    model_version: str
    inference_duration_ms: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    is_ready: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
