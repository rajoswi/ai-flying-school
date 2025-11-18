
# train.py - trains a RandomForest on the sample CSV and saves model.joblib
import pandas as pd
from model import build_pipeline
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

DF = pd.read_csv('cadet_weekly.csv')

def label_row(r):
    if r['incident_reports']>0 or r['attendance_pct']<50 or r['sim_performance_score']<50:
        return 'High'
    if r['attendance_pct']<75 or r['ground_exam_score']<60 or r['late_count']>2:
        return 'Medium'
    return 'Low'

DF['risk_label'] = DF.apply(label_row, axis=1)
DF['risk_num'] = DF['risk_label'].map({'Low':0,'Medium':1,'High':2})

FEATURES = ['attendance_pct','flight_hours_week','total_flight_hours','sim_performance_score','ground_exam_score','late_count','uniform_failures','incident_reports','instructor_notes_len']
X = DF[FEATURES]
y = DF['risk_num']

if len(DF) < 10:
    raise SystemExit('Not enough data for training in cadet_weekly.csv')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

pipe = build_pipeline()
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
print(classification_report(y_test,pred))

dump(pipe,'model.joblib')
print('Saved model.joblib')
