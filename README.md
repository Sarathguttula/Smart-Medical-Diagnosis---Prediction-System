# Smart Medical Diagnosis & Prediction System

An educational prototype for symptom-based disease prediction, risk scoring, recommendations, dashboard visualization, and model explainability.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Start the API: `uvicorn backend.app.main:app --reload`
4. Open http://127.0.0.1:8000

The API docs are available at http://127.0.0.1:8000/docs.

## API

- `GET /api/symptoms` returns supported symptoms.
- `POST /api/predict` accepts `{ "symptoms": ["fever", "cough"] }`.
- `GET /api/health` checks service status.

The training data is synthetic and the results must not be used for diagnosis, treatment, or emergency decisions. Seek advice from a qualified healthcare professional.
