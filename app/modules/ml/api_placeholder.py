"""
ML API Integration Placeholder
================================
This module is reserved for future ML prediction API endpoints.

When prediction endpoints are ready, create:

    app/modules/ml/routes.py

And register in app/api/v1/router.py:

    from app.modules.ml.routes import router as ml_router
    api_router.include_router(ml_router)

Planned endpoints:
    POST /ml/predict        → single prediction
    POST /ml/predict/batch  → batch predictions
    GET  /ml/models         → list available model versions
    GET  /ml/models/{name}  → model metadata and metrics

The Predictor and PredictionFormatter classes in this module
are designed to plug directly into those routes.
"""
