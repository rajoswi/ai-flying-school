
# model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['attendance_pct','flight_hours_week','total_flight_hours','sim_performance_score','ground_exam_score','late_count','uniform_failures','incident_reports','instructor_notes_len']

def build_pipeline():
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf),
    ])
    return pipe

def predict_risk(pipe, X_df):
    probs = pipe.predict_proba(X_df)
    classes = pipe.named_steps['clf'].classes_
    idx_map = {c:i for i,c in enumerate(classes)}
    p_med = probs[:, idx_map.get(1,0)] if 1 in idx_map else np.zeros(len(probs))
    p_high = probs[:, idx_map.get(2,0)] if 2 in idx_map else np.zeros(len(probs))
    risk_score = 0.5 * p_med + 1.0 * p_high
    pred_class = pipe.predict(X_df)
    labels = ['Low','Medium','High']
    pred_label = [labels[int(c)] for c in pred_class]
    return risk_score, pred_label
