# backend/model_training.py
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import os
import tensorflow as tf                
from tensorflow.keras.models import load_model  

# 常量定义
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
        """只加载预训练模型,不再训练"""
        xgb_path = os.path.join(MODEL_DIR, 'xgb_model1.joblib')
        dnn_path = os.path.join(MODEL_DIR, 'dnn_model.keras')
        preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.joblib')
        
        print(f"🔍 Looking for models in: {MODEL_DIR}")
        print(f"   XGBoost exists: {os.path.exists(xgb_path)}")
        print(f"   DNN exists: {os.path.exists(dnn_path)}")
        print(f"   Preprocessor exists: {os.path.exists(preprocessor_path)}")
        
        if os.path.exists(xgb_path) and os.path.exists(dnn_path) and os.path.exists(preprocessor_path):
            try:
                print("✅ Loading pre-trained models...")
                
                self.models['xgboost'] = joblib.load(xgb_path)
                print("   ✓ XGBoost loaded")
                
                self.models['dnn'] = tf.keras.models.load_model(dnn_path)
                print("   ✓ DNN loaded")
                
                self.preprocessor = joblib.load(preprocessor_path)
                print("   ✓ Preprocessor loaded")
                
                self.is_trained = True
                print("🎉 All models loaded successfully!")
                return
                
            except Exception as e:
                print(f"❌ Error loading models: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to load pre-trained models: {e}")
        else:
            missing_files = []
            if not os.path.exists(xgb_path):
                missing_files.append("xgb_model1.joblib")
            if not os.path.exists(dnn_path):
                missing_files.append("dnn_model.keras")
            if not os.path.exists(preprocessor_path):
                missing_files.append("preprocessor.joblib")
            
            error_msg = f"""
❌ Pre-trained models not found!
Missing files: {', '.join(missing_files)}
Expected location: {MODEL_DIR}

Files in directory: {os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else 'Directory not found'}

Please ensure these files are committed to Git:
  1. backend/models/xgb_model1.joblib
  2. backend/models/dnn_model.keras
  3. backend/models/preprocessor.joblib
            """
            print(error_msg)
            raise FileNotFoundError(error_msg)
    
    def preprocess_user_data(self, user_data):

        try:
            print("\n" + "="*60)
            print("🔄 PREPROCESSING USER DATA")
            print("="*60)
            print(f"📥 Received data keys: {list(user_data.keys())}")
            
            processed_data = {}
            
            # Step 1: Diabetes Basics
            processed_data['age'] = float(user_data.get('age', 50))
            processed_data['diabetes_type'] = user_data.get('diabetes_type', 'Type 2')
            processed_data['duration'] = float(user_data.get('duration', 5))
            processed_data['hba1c'] = float(user_data.get('hba1c', 7.0))
            processed_data['fbg'] = float(user_data.get('fbg', 120))
            
            processed_data['Management_Insulin'] = int(user_data.get('Management_Insulin', 0))
            processed_data['Management_OralMed'] = int(user_data.get('Management_OralMed', 0))
            processed_data['Management_DietExercise'] = int(user_data.get('Management_DietExercise', 0))
            processed_data['blood_sugar_frequency'] = user_data.get('blood_sugar_frequency', 'Once a day')
            
            # Step 2: Health Conditions
            processed_data['hypertension'] = int(user_data.get('hypertension', 0))
            processed_data['systolic_bp'] = float(user_data.get('systolic_bp', 120))
            processed_data['diastolic_bp'] = float(user_data.get('diastolic_bp', 80))
            processed_data['nephropathy'] = int(user_data.get('nephropathy', 0))
            processed_data['neuropathy'] = int(user_data.get('neuropathy', 0))
            processed_data['cholesterol'] = int(user_data.get('cholesterol', 0))
            
            # Step 3: Lifestyle
            processed_data['smoking'] = user_data.get('smoking', 'Never smoked')
            processed_data['Height_cm'] = float(user_data.get('Height_cm', 170))
            processed_data['Weight_kg'] = float(user_data.get('Weight_kg', 70))
            processed_data['BMI'] = float(user_data.get('BMI', 24.2))
            
            # Step 4: Eye Health
            processed_data['last_eye_exam'] = user_data.get('last_eye_exam', 'More than 2 years ago')
            processed_data['diagnosed_retinopathy'] = int(user_data.get('diagnosed_retinopathy', 0))
            processed_data['Vision_Blurriness'] = int(user_data.get('Vision_Blurriness', 0))
            processed_data['Vision_Floaters'] = int(user_data.get('Vision_Floaters', 0))
            processed_data['Vision_Fluctuating'] = int(user_data.get('Vision_Fluctuating', 0))
            processed_data['Vision_Sudden_Loss'] = int(user_data.get('Vision_Sudden_Loss', 0))
            processed_data['medication_adherence'] = user_data.get('medication_adherence', 'I never miss a dose')
            
            print(f"✅ Processed sample:")
            print(f"   Age: {processed_data['age']}, HbA1c: {processed_data['hba1c']}, BMI: {processed_data['BMI']:.1f}")
            
            df_user = pd.DataFrame([processed_data])
            
            expected_columns = [
                'age', 'diabetes_type', 'duration', 'hba1c', 'fbg',
                'Management_Insulin', 'Management_OralMed', 'Management_DietExercise',
                'blood_sugar_frequency', 'hypertension', 'systolic_bp', 'diastolic_bp',
                'nephropathy', 'neuropathy', 'cholesterol', 'smoking',
                'Height_cm', 'Weight_kg', 'BMI', 'last_eye_exam',
                'diagnosed_retinopathy', 'Vision_Blurriness', 'Vision_Floaters',
                'Vision_Fluctuating', 'Vision_Sudden_Loss', 'medication_adherence'
            ]
            
            df_user = df_user[expected_columns]
            features = self.preprocessor.transform(df_user)
            
            print(f"✅ Final feature shape: {features.shape}")
            print("="*60 + "\n")
            
            return features
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            
            default_df = pd.DataFrame([{
                'age': 50, 'diabetes_type': 'Type 2', 'duration': 5,
                'hba1c': 7.0, 'fbg': 120,
                'Management_Insulin': 0, 'Management_OralMed': 1, 'Management_DietExercise': 0,
                'blood_sugar_frequency': 'Once a day', 'hypertension': 0,
                'systolic_bp': 120, 'diastolic_bp': 80,
                'nephropathy': 0, 'neuropathy': 0, 'cholesterol': 0,
                'smoking': 'Never smoked', 'Height_cm': 170, 'Weight_kg': 70, 'BMI': 24.2,
                'last_eye_exam': '1-2 years ago', 'diagnosed_retinopathy': 0,
                'Vision_Blurriness': 0, 'Vision_Floaters': 0,
                'Vision_Fluctuating': 0, 'Vision_Sudden_Loss': 0,
                'medication_adherence': 'I never miss a dose'
            }])
            return self.preprocessor.transform(default_df)
    
    def predict_risk(self, user_data):
        """预测风险(XGB × DNN ensemble)"""
        try:
            features = self.preprocess_user_data(user_data)
            
            xgb_prob = self.models['xgboost'].predict_proba(features)[:, 1][0]
            dnn_prob = self.models['dnn'].predict(features, verbose=0)[0, 0]
            
            final_probability = ensemble_average(xgb_prob, dnn_prob, self.optimal_weight)
            risk_score = int(final_probability * 100)
            
            if risk_score >= 70:
                risk_level = "High Risk"
            elif risk_score >= 30:
                risk_level = "Moderate Risk"
            else:
                risk_level = "Low Risk"
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'probability': round(float(final_probability), 3)
            }
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'risk_level': "Low Risk",
                'risk_score': 15,
                'probability': 0.15
            }
    
    def explain_prediction(self, user_data):
        """解释预测结果"""
        try:
            features = self.preprocess_user_data(user_data)[0]
            explanations = []
            
            age = features[0]
            if age > 60:
                explanations.append({
                    'factor': 'Age',
                    'impact': 'High',
                    'explanation': f'Age {int(age)} increases retinopathy risk'
                })
            elif age > 40:
                explanations.append({
                    'factor': 'Age',
                    'impact': 'Medium',
                    'explanation': f'Age {int(age)} moderately affects risk'
                })
            
            duration = features[1]
            if duration > 10:
                explanations.append({
                    'factor': 'Diabetes Duration',
                    'impact': 'High',
                    'explanation': f'{int(duration)} years with diabetes significantly increases risk'
                })
            elif duration > 5:
                explanations.append({
                    'factor': 'Diabetes Duration',
                    'impact': 'Medium',
                    'explanation': f'{int(duration)} years with diabetes moderately increases risk'
                })
            
            hba1c = features[2]
            if hba1c > 8.0:
                explanations.append({
                    'factor': 'Blood Sugar Control (HbA1c)',
                    'impact': 'High',
                    'explanation': f'HbA1c level of {hba1c:.1f}% indicates poor glucose control'
                })
            elif hba1c > 7.0:
                explanations.append({
                    'factor': 'Blood Sugar Control (HbA1c)',
                    'impact': 'Medium',
                    'explanation': f'HbA1c level of {hba1c:.1f}% needs improvement'
                })
            
            fbg = features[3]
            if fbg > 180:
                explanations.append({
                    'factor': 'Fasting Blood Glucose',
                    'impact': 'High',
                    'explanation': f'Fasting blood glucose of {fbg:.0f} mg/dL indicates poor diabetes control'
                })
            elif fbg > 130:
                explanations.append({
                    'factor': 'Fasting Blood Glucose',
                    'impact': 'Medium',
                    'explanation': f'Fasting blood glucose of {fbg:.0f} mg/dL needs improvement'
                })
            
            if len(features) > 4 and features[4] == 1.0:
                explanations.append({
                    'factor': 'High Blood Pressure',
                    'impact': 'Medium',
                    'explanation': 'Hypertension can accelerate retinopathy development'
                })
            
            return explanations[:5]
            
        except Exception as e:
            print(f"❌ Explanation error: {e}")
            return [{
                'factor': 'Basic Assessment',
                'impact': 'Low',
                'explanation': 'Standard diabetes management recommended'
            }]

if __name__ == "__main__":
    print("Testing DRRiskModel...")
    model = DRRiskModel()
    
    test_data = {
        'age': 55,
        'diabetes_type': 'Type 2',
        'duration': 12,
        'hba1c': 8.5,
        'fbg': 180,
        'Management_Insulin': 1,
        'Management_OralMed': 1,
        'Management_DietExercise': 0,
        'blood_sugar_frequency': 'Once a day',
        'hypertension': 1,
        'systolic_bp': 140,
        'diastolic_bp': 90,
        'nephropathy': 0,
        'neuropathy': 0,
        'cholesterol': 1,
        'smoking': 'Former smoker',
        'Height_cm': 170,
        'Weight_kg': 85,
        'BMI': 29.4,
        'last_eye_exam': 'More than 2 years ago',
        'diagnosed_retinopathy': 0,
        'Vision_Blurriness': 1,
        'Vision_Floaters': 0,
        'Vision_Fluctuating': 0,
        'Vision_Sudden_Loss': 0,
        'medication_adherence': 'I occasionally miss doses'
    }
    
    prediction = model.predict_risk(test_data)
    print("\n📊 Prediction Results:")
    print(f"  Risk Level: {prediction['risk_level']}")
    print(f"  Risk Score: {prediction['risk_score']}/100")
    print(f"  Probability: {prediction['probability']}")