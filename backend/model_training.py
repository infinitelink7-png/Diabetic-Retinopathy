# backend/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import os

class DRRiskModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        
    def create_model(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    
    def prepare_features(self, user_data):
        """Convert frontend data to model features"""
        features = {}
        
        # Basic features
        features['age'] = int(user_data.get('age', 50))
        features['diabetes_type'] = user_data.get('diabetes_type', 'Type 2')
        features['diabetes_duration'] = int(user_data.get('duration', 5))
        
        # HbA1c processing (based on your radio buttons)
        hba1c_choice = user_data.get('hba1c', 'Normal')
        if hba1c_choice == 'Normal':
            features['hba1c'] = 5.0
        elif hba1c_choice == 'Prediabetes':
            features['hba1c'] = 6.0
        else:  # Type 2 Diabetes
            features['hba1c'] = 7.5
        
        # Blood pressure related
        features['hypertension'] = user_data.get('hypertension', 'No')
        features['systolic_bp'] = int(user_data.get('bp_sys', 120))
        features['diastolic_bp'] = int(user_data.get('bp_dia', 80))
        
        # Other health indicators
        features['nephropathy'] = user_data.get('nephropathy', 'No')
        features['neuropathy'] = user_data.get('neuropathy', 'No')
        features['high_cholesterol'] = user_data.get('cholesterol', 'No')
        
        # Lifestyle
        features['smoking_status'] = user_data.get('smoking', 'Never Smoked')
        
        # BMI
        features['bmi'] = float(user_data.get('bmi', 25.0))
        
        # Eye health related
        features['last_eye_exam'] = user_data.get('eye_exam', 'Never')
        features['previous_dr_diagnosis'] = user_data.get('dr_diag', 'No')
        
        # Medication adherence
        adherence_map = {
            'I never miss a dose': 'Never miss',
            'I occasionally miss doses': 'Occasionally miss', 
            'I often forget to take it': 'Often forget',
            'I do not take any medication': 'No medication'
        }
        features['medication_adherence'] = adherence_map.get(
            user_data.get('adherence', 'I never miss a dose'), 'Never miss'
        )
        
        return pd.DataFrame([features])
    
    def predict_risk(self, user_data):
        """Predict diabetic retinopathy risk"""
        try:
            # Prepare features
            features_df = self.prepare_features(user_data)
            
            # If no trained model, use rule-based prediction
            if self.model is None:
                return self.rule_based_prediction(features_df.iloc[0])
            
            # Preprocess features
            processed_features = self.preprocessor.transform(features_df)
            
            # Predict
            probability = self.model.predict_proba(processed_features)[0]
            risk_score = int(probability[1] * 100)
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = "High Risk"
            elif risk_score >= 30:
                risk_level = "Moderate Risk" 
            else:
                risk_level = "Low Risk"
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'probability': float(probability[1])
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.rule_based_prediction(user_data)
    
    def rule_based_prediction(self, features):
        """Rule-based prediction (fallback solution)"""
        risk_score = 0
        
        # HbA1c risk
        hba1c = features.get('hba1c', 5.0)
        if hba1c > 8.0:
            risk_score += 30
        elif hba1c > 6.5:
            risk_score += 15
        
        # Diabetes duration risk
        duration = features.get('diabetes_duration', 0)
        if duration > 15:
            risk_score += 25
        elif duration > 10:
            risk_score += 15
        elif duration > 5:
            risk_score += 10
        
        # Hypertension risk
        if features.get('hypertension') == 'Yes':
            risk_score += 15
        
        # Kidney disease risk
        if features.get('nephropathy') == 'Yes':
            risk_score += 20
        
        # Previous diagnosis risk
        if features.get('previous_dr_diagnosis') == 'Yes':
            risk_score += 25
        
        # Smoking risk
        if features.get('smoking_status') == 'Current Smoker':
            risk_score += 10
        
        # Limit risk score between 0-100
        risk_score = min(100, max(0, risk_score))
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "High Risk"
        elif risk_score >= 30:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'probability': risk_score / 100.0
        }
    
    def explain_prediction(self, user_data):
        """Explain prediction results"""
        features = self.prepare_features(user_data).iloc[0]
        factors = []
        
        # Analyze main risk factors
        if features.get('hba1c', 0) > 7.0:
            factors.append({
                'factor': 'Blood Sugar Control',
                'impact': 'High',
                'description': f'Higher HbA1c level ({features["hba1c"]}%)'
            })
        
        if features.get('diabetes_duration', 0) > 10:
            factors.append({
                'factor': 'Diabetes Duration', 
                'impact': 'High',
                'description': f'Longer diabetes duration ({features["diabetes_duration"]} years)'
            })
        
        if features.get('hypertension') == 'Yes':
            factors.append({
                'factor': 'High Blood Pressure',
                'impact': 'Medium',
                'description': 'History of hypertension'
            })
        
        if features.get('previous_dr_diagnosis') == 'Yes':
            factors.append({
                'factor': 'Previous Diagnosis',
                'impact': 'High', 
                'description': 'Previously diagnosed with diabetic retinopathy'
            })
        
        if features.get('nephropathy') == 'Yes':
            factors.append({
                'factor': 'Kidney Disease',
                'impact': 'High',
                'description': 'Diagnosed with diabetic kidney disease'
            })
        
        if features.get('smoking_status') == 'Current Smoker':
            factors.append({
                'factor': 'Smoking',
                'impact': 'Medium',
                'description': 'Current smoking status'
            })
        
        # If no specific factors identified, provide general advice
        if not factors:
            factors.append({
                'factor': 'Routine Monitoring',
                'impact': 'Low',
                'description': 'Recommend continuing regular eye examinations'
            })
        
        return factors
    
    def save_model(self, filepath):
        """Save model"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'preprocessor': self.preprocessor
            }, filepath)
    
    def load_model(self, filepath):
        """Load model"""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data['model']
            self.preprocessor = data['preprocessor']