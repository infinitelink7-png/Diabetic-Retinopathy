# backend/model_training.py
import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import load_model
import traceback

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def ensemble_average(prob1, prob2, weight1=0.5):
    return weight1 * prob1 + (1 - weight1) * prob2

class DRRiskModel:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.is_trained = False
        self.optimal_weight = 0.5
        self.load_or_train_models()

    def load_or_train_models(self):
        xgb_path = os.path.join(MODEL_DIR, 'xgb_model1.joblib')
        dnn_path_h5 = os.path.join(MODEL_DIR, 'dnn_model.h5')
        dnn_path_keras = os.path.join(MODEL_DIR, 'dnn_model.keras')
        preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.joblib')

        print(f"🔍 Looking for models in: {MODEL_DIR}")
        print(f"   XGBoost exists: {os.path.exists(xgb_path)}")
        print(f"   DNN (.h5) exists: {os.path.exists(dnn_path_h5)}")
        print(f"   DNN (.keras) exists: {os.path.exists(dnn_path_keras)}")
        print(f"   Preprocessor exists: {os.path.exists(preprocessor_path)}")

        if os.path.exists(xgb_path) and (os.path.exists(dnn_path_h5) or os.path.exists(dnn_path_keras)) and os.path.exists(preprocessor_path):
            try:
                print("✅ Loading pre-trained models...")
                self.models['xgboost'] = joblib.load(xgb_path)
                print("   ✓ XGBoost loaded")
                if os.path.exists(dnn_path_h5):
                    self.models['dnn'] = tf.keras.models.load_model(dnn_path_h5, compile=False)
                    print("   ✓ DNN (.h5) loaded successfully")
                else:
                    self.models['dnn'] = tf.keras.models.load_model(dnn_path_keras, compile=False)
                    print("   ✓ DNN (.keras) loaded successfully")
                self.preprocessor = joblib.load(preprocessor_path)
                print("   ✓ Preprocessor loaded")
                self.is_trained = True
                print("🎉 All models loaded successfully!")
                return
            except Exception as e:
                print(f"❌ Error loading models: {e}")
                traceback.print_exc()
                raise RuntimeError(f"Failed to load pre-trained models: {e}")
        else:
            missing_files = []
            if not os.path.exists(xgb_path):
                missing_files.append("xgb_model1.joblib")
            if not (os.path.exists(dnn_path_h5) or os.path.exists(dnn_path_keras)):
                missing_files.append("dnn_model.h5 or dnn_model.keras")
            if not os.path.exists(preprocessor_path):
                missing_files.append("preprocessor.joblib")
            error_msg = f"""
❌ Pre-trained models not found!
Missing files: {', '.join(missing_files)}
Expected location: {MODEL_DIR}

Files in directory: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}
"""
            print(error_msg)
            raise FileNotFoundError(error_msg)

    def preprocess_user_data(self, user_data):
        try:
            print("\n" + "="*60)
            print("🔄 PREPROCESSING USER DATA")
            print("="*60)
            print(f"📥 Received data keys: {list(user_data.keys())}")

            processed_data = {
                'age': float(user_data.get('age', 50)),
                'diabetes_type': user_data.get('diabetes_type', 'Type 2'),
                'duration': float(user_data.get('duration', 5)),
                'hba1c': float(user_data.get('hba1c', 7.0)),
                'fbg': float(user_data.get('fbg', 120)),
                'Management_Insulin': int(user_data.get('Management_Insulin', 0)),
                'Management_OralMed': int(user_data.get('Management_OralMed', 0)),
                'Management_DietExercise': int(user_data.get('Management_DietExercise', 0)),
                'blood_sugar_frequency': user_data.get('blood_sugar_frequency', 'Once a day'),
                'hypertension': int(user_data.get('hypertension', 0)),
                'systolic_bp': float(user_data.get('systolic_bp', 120)),
                'diastolic_bp': float(user_data.get('diastolic_bp', 80)),
                'nephropathy': int(user_data.get('nephropathy', 0)),
                'neuropathy': int(user_data.get('neuropathy', 0)),
                'cholesterol': int(user_data.get('cholesterol', 0)),
                'smoking': user_data.get('smoking', 'Never smoked'),
                'Height_cm': float(user_data.get('Height_cm', 170)),
                'Weight_kg': float(user_data.get('Weight_kg', 70)),
                'BMI': float(user_data.get('BMI', 24.2)),
                'last_eye_exam': user_data.get('last_eye_exam', 'More than 2 years ago'),
                'diagnosed_retinopathy': int(user_data.get('diagnosed_retinopathy', 0)),
                'Vision_Blurriness': int(user_data.get('Vision_Blurriness', 0)),
                'Vision_Floaters': int(user_data.get('Vision_Floaters', 0)),
                'Vision_Fluctuating': int(user_data.get('Vision_Fluctuating', 0)),
                'Vision_Sudden_Loss': int(user_data.get('Vision_Sudden_Loss', 0)),
                'medication_adherence': user_data.get('medication_adherence', 'I never miss a dose')
            }

            df_user = pd.DataFrame([processed_data])
            features = self.preprocessor.transform(df_user)
            print(f"✅ Final feature shape: {features.shape}")
            print("="*60 + "\n")
            return features
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            traceback.print_exc()
            raise e

    def predict_risk(self, user_data):
        try:
            features = self.preprocess_user_data(user_data)
            selected_model = (user_data.get('selected_model') or 'ensemble').lower()
            print(f"Using model: {selected_model}")

            if selected_model == 'xgboost':
                xgb_prob = float(self.models['xgboost'].predict_proba(features)[:, 1][0])
                final_probability = xgb_prob
                model_used = 'xgboost'

            elif selected_model == 'dnn':
                dnn_prob = float(self.models['dnn'].predict(features, verbose=0).ravel()[0])
                final_probability = dnn_prob
                model_used = 'dnn'

            else:
                xgb_prob = float(self.models['xgboost'].predict_proba(features)[:, 1][0])
                dnn_prob = float(self.models['dnn'].predict(features, verbose=0).ravel()[0])
                final_probability = ensemble_average(xgb_prob, dnn_prob, self.optimal_weight)
                model_used = 'ensemble'

            risk_score = int(round(final_probability * 100))
            if risk_score >= 70:
                risk_level = "High Risk"
            elif risk_score >= 30:
                risk_level = "Moderate Risk"
            else:
                risk_level = "Low Risk"

            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'probability': round(float(final_probability), 3),
                'model_used': model_used
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            return {
                'risk_level': "Unknown",
                'risk_score': 0,
                'probability': 0.0,
                'model_used': user_data.get('selected_model', 'ensemble')
            }

    def explain_prediction(self, user_data):
        """Analyze key factors directly from original input data (not feature indices)."""
        try:
            explanations = []

            age = float(user_data.get('age', 50))
            duration = float(user_data.get('duration', 5))
            hba1c = float(user_data.get('hba1c', 7.0))
            fbg = float(user_data.get('fbg', 120))
            hypertension = int(user_data.get('hypertension', 0))
            nephropathy = int(user_data.get('nephropathy', 0))
            neuropathy = int(user_data.get('neuropathy', 0))
            cholesterol = int(user_data.get('cholesterol', 0))
            bmi = float(user_data.get('BMI', 24.2))

            if age > 60:
                explanations.append({'factor': 'Age', 'impact': 'High', 'explanation': f'Age {int(age)} increases retinopathy risk'})
            elif age > 40:
                explanations.append({'factor': 'Age', 'impact': 'Medium', 'explanation': f'Age {int(age)} moderately affects risk'})

            if duration > 10:
                explanations.append({'factor': 'Diabetes Duration', 'impact': 'High', 'explanation': f'{int(duration)} years with diabetes significantly increases risk'})
            elif duration > 5:
                explanations.append({'factor': 'Diabetes Duration', 'impact': 'Medium', 'explanation': f'{int(duration)} years with diabetes moderately increases risk'})

            if hba1c > 8.0:
                explanations.append({'factor': 'Blood Sugar (HbA1c)', 'impact': 'High', 'explanation': f'HbA1c {hba1c:.1f}% indicates poor glucose control'})
            elif hba1c > 7.0:
                explanations.append({'factor': 'Blood Sugar (HbA1c)', 'impact': 'Medium', 'explanation': f'HbA1c {hba1c:.1f}% needs improvement'})

            if fbg > 180:
                explanations.append({'factor': 'Fasting Blood Glucose', 'impact': 'High', 'explanation': f'FBG {fbg:.0f} mg/dL indicates poor diabetes control'})
            elif fbg > 130:
                explanations.append({'factor': 'Fasting Blood Glucose', 'impact': 'Medium', 'explanation': f'FBG {fbg:.0f} mg/dL needs improvement'})
            elif fbg < 70:
                explanations.append({'factor': 'Fasting Blood Glucose', 'impact': 'Medium', 'explanation': f'FBG {fbg:.0f} mg/dL may be too low'})

            if hypertension == 1:
                explanations.append({'factor': 'High Blood Pressure', 'impact': 'Medium', 'explanation': 'Hypertension can accelerate retinopathy development'})

            if nephropathy == 1:
                explanations.append({'factor': 'Kidney Disease', 'impact': 'High', 'explanation': 'Diabetic kidney disease is closely linked to retinopathy'})

            if neuropathy == 1:
                explanations.append({'factor': 'Nerve Damage', 'impact': 'Medium', 'explanation': 'Diabetic neuropathy may indicate systemic complications'})

            if cholesterol == 1:
                explanations.append({'factor': 'High Cholesterol', 'impact': 'Low', 'explanation': 'High cholesterol can contribute to vascular complications'})

            if bmi > 30:
                explanations.append({'factor': 'Body Mass Index', 'impact': 'Medium', 'explanation': f'BMI {bmi:.1f} indicates obesity, which can worsen diabetes control'})
            elif bmi > 25:
                explanations.append({'factor': 'Body Mass Index', 'impact': 'Low', 'explanation': f'BMI {bmi:.1f} indicates overweight status'})

            return explanations[:5]

        except Exception as e:
            print(f"❌ Explanation error: {e}")
            traceback.print_exc()
            return [{
                'factor': 'Basic Assessment',
                'impact': 'Low',
                'explanation': 'Standard diabetes management recommended'
            }]

if __name__ == "__main__":
    print("Testing DRRiskModel...")
    model = DRRiskModel()
    test_data = {
        'age': 55, 'diabetes_type': 'Type 2', 'duration': 12, 'hba1c': 8.5, 'fbg': 180,
        'Management_Insulin': 1, 'Management_OralMed': 1, 'Management_DietExercise': 0,
        'blood_sugar_frequency': 'Once a day', 'hypertension': 1, 'systolic_bp': 140,
        'diastolic_bp': 90, 'nephropathy': 0, 'neuropathy': 0, 'cholesterol': 1,
        'smoking': 'Former smoker', 'Height_cm': 170, 'Weight_kg': 85, 'BMI': 29.4,
        'last_eye_exam': 'More than 2 years ago', 'diagnosed_retinopathy': 0,
        'Vision_Blurriness': 1, 'Vision_Floaters': 0, 'Vision_Fluctuating': 0,
        'Vision_Sudden_Loss': 0, 'medication_adherence': 'I occasionally miss doses'
    }
    prediction = model.predict_risk(test_data)
    print("\n📊 Prediction Results:")
    print(f"  Risk Level: {prediction['risk_level']}")
    print(f"  Risk Score: {prediction['risk_score']}/100")
    print(f"  Probability: {prediction['probability']}")
    print("\n🔍 Key Factors:")
    print(model.explain_prediction(test_data))
