
# app.py - FastAPI app
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from joblib import load
from model import FEATURES, predict_risk
import io

app = FastAPI(title='AI Cadet Risk Detector')

# load model at startup if available
try:
    MODEL = load('model.joblib')
except Exception as e:
    MODEL = None
    print('Model not loaded:', e)

class CadetInput(BaseModel):
    cadet_id: str
    name: str
    batch: str
    attendance_pct: float
    flight_hours_week: float
    total_flight_hours: float
    sim_performance_score: float
    ground_exam_score: float
    late_count: int
    uniform_failures: int
    incident_reports: int
    instructor_notes_len: int

@app.post('/predict_single')
async def predict_single(payload: CadetInput):
    if MODEL is None:
        return {'error':'Model not trained. Run train.py first.'}
    df = pd.DataFrame([payload.dict()])
    X = df[FEATURES]
    risk_score, pred_label = predict_risk(MODEL, X)
    return {
        'cadet_id': payload.cadet_id,
        'name': payload.name,
        'risk_score': float(risk_score[0]) if hasattr(risk_score,'__len__') else float(risk_score),
        'risk_label': pred_label[0]
    }

@app.post('/predict_csv')
async def predict_csv(file: UploadFile = File(...)):
    if MODEL is None:
        return {'error':'Model not trained. Run train.py first.'}
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    X = df[FEATURES]
    risk_score, pred_label = predict_risk(MODEL, X)
    df['risk_score'] = list(map(float, risk_score))
    df['risk_label'] = pred_label
    return df.to_dict(orient='records')

@app.get('/health')
async def health():
    return {'status':'ok'}
