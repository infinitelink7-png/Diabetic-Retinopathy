# backend/model_training.py
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

class DRRiskModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'diabetes_duration', 'hba1c_level', 'fbg_level', 'has_hypertension',
            'has_nephropathy', 'has_neuropathy', 'has_high_cholesterol',
            'bmi', 'is_smoker', 'years_since_last_eye_exam', 'medication_adherence'
        ]
        self.is_trained = False
        self.load_or_train_models()
    
    def preprocess_user_data(self, user_data):
        """Preprocess user data from frontend for model prediction"""
        try:
            features = np.zeros(len(self.feature_names))
            
            # 1. Age (索引0)
            features[0] = float(user_data.get('age', 50))
            
            # 2. Diabetes duration (索引1)
            features[1] = float(user_data.get('duration', 5))
            
            # 3. HbA1c level (索引2)
            hba1c_map = {'Normal': 5.0, 'Prediabetes': 6.0, 'Type 2 Diabetes': 7.5}
            hba1c_value = user_data.get('hba1c', 'Normal')
            features[2] = hba1c_map.get(hba1c_value, 6.0)

            # 4. Fasting Blood Glucose (索引3)
            fbg = user_data.get('fbg', '120')
            features[3] = float(fbg) if fbg else 120.0
            
            # 5. Hypertension (索引4)
            features[4] = 1.0 if user_data.get('hypertension') == 'Yes' else 0.0
            
            # 6. Nephropathy (索引5)
            features[5] = 1.0 if user_data.get('nephropathy') == 'Yes' else 0.0
            
            # 7. Neuropathy (索引6)
            features[6] = 1.0 if user_data.get('neuropathy') == 'Yes' else 0.0
            
            # 8. High cholesterol (索引7)
            features[7] = 1.0 if user_data.get('cholesterol') == 'Yes' else 0.0
            
            # 9. BMI (索引8)
            bmi = user_data.get('bmi', 25.0)
            if isinstance(bmi, str):
                bmi = float(bmi) if bmi.replace('.', '').isdigit() else 25.0
            features[8] = float(bmi)
            
            # 10. Smoking status (索引9)
            smoking_status = user_data.get('smoking', 'Never')
            features[9] = 1.0 if smoking_status == 'Current' else 0.0
            
            # 11. Years since last eye exam (索引10)
            eye_exam_map = {
                'Within the past year': 0.5,
                '1-2 years ago': 1.5,
                '3-5 years ago': 4.0,
                'Over 5 years ago': 7.0,
                'Never': 10.0
            }
            features[10] = eye_exam_map.get(user_data.get('eye_exam', 'Never'), 5.0)
            
            # 12. Medication adherence (索引11)
            adherence_map = {
                'I never miss a dose': 1.0,
                'I rarely miss a dose': 0.8,
                'I sometimes miss a dose': 0.5,
                'I often miss doses': 0.2
            }
            features[11] = adherence_map.get(user_data.get('adherence', 'I sometimes miss a dose'), 0.5)
            
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return np.array([[50, 5, 6.0, 120.0, 0, 0, 0, 0, 25.0, 0, 5.0, 0.5]])
    
    def create_training_data(self):
        """Create synthetic training data based on clinical knowledge"""
        np.random.seed(42)
        n_samples = 5000
        
        data = []
        for i in range(n_samples):
            # Generate realistic patient data
            age = np.random.normal(58, 12)
            age = max(20, min(90, age))
            
            diabetes_duration = np.random.exponential(8)
            diabetes_duration = min(50, diabetes_duration)
            
            # HbA1c - higher for diabetic patients
            hba1c = np.random.normal(7.5, 1.5) if np.random.random() > 0.3 else np.random.normal(5.8, 0.5)
            hba1c = max(4.0, min(15.0, hba1c))

            # Fasting Blood Glucose 
            fbg = np.random.normal(140, 30) if hba1c > 6.5 else np.random.normal(100, 15)
            fbg = max(70, min(300, fbg))
            
            # Comorbidities based on clinical risk factors
            hypertension_prob = 0.3 + (age - 50) * 0.01 + diabetes_duration * 0.02
            has_hypertension = np.random.random() < min(0.9, hypertension_prob)
            
            nephropathy_prob = 0.1 + diabetes_duration * 0.03 + (hba1c - 6) * 0.05
            has_nephropathy = np.random.random() < min(0.7, nephropathy_prob)
            
            neuropathy_prob = 0.1 + diabetes_duration * 0.02
            has_neuropathy = np.random.random() < min(0.6, neuropathy_prob)
            
            high_cholesterol_prob = 0.25 + (age - 50) * 0.008
            has_high_cholesterol = np.random.random() < min(0.8, high_cholesterol_prob)
            
            bmi = np.random.normal(28, 6)
            bmi = max(18, min(45, bmi))
            
            is_smoker = np.random.random() < 0.15
            
            years_since_eye_exam = np.random.exponential(2)
            years_since_eye_exam = min(15, years_since_eye_exam)
            
            medication_adherence = np.random.beta(2, 2)
            
            # Calculate DR risk based on clinical factors
            base_risk = 0.05
            risk_factors = (
                (diabetes_duration / 10) * 0.20 +
                max(0, (hba1c - 6.5) / 3) * 0.18 +
                max(0, (fbg - 100) / 50) * 0.15 +
                (1 if has_hypertension else 0) * 0.12 +
                (1 if has_nephropathy else 0) * 0.15 +
                (1 if has_high_cholesterol else 0) * 0.08 +
                max(0, (bmi - 25) / 20) * 0.08 +
                (1 if is_smoker else 0) * 0.12 +
                (years_since_eye_exam / 5) * 0.08 +
                (1 - medication_adherence) * 0.08
            )
            
            dr_probability = min(0.95, base_risk + risk_factors * 0.8)
            has_dr = np.random.random() < dr_probability
            
            data.append([
                age, diabetes_duration, hba1c, fbg, has_hypertension,
                has_nephropathy, has_neuropathy, has_high_cholesterol,
                bmi, is_smoker, years_since_eye_exam, medication_adherence,
                has_dr
            ])
        
        df = pd.DataFrame(data, columns=self.feature_names + ['has_diabetic_retinopathy'])
        return df
    
    def build_xgboost_model(self):
        """Build and train XGBoost model"""
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        return model
    
    def build_dnn_model(self, input_dim):
        """Build Deep Neural Network model"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_1d_cnn_model(self, input_dim):
        """Build 1D CNN model"""
        model = keras.Sequential([
            layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_models(self):
        """Train all three models"""
        print("Generating training data...")
        df = self.create_training_data()
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['has_diabetic_retinopathy']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"DR prevalence: {y.mean():.2%}")
        
        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = self.build_xgboost_model()
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgboost'] = xgb_model
        
        # Train DNN
        print("Training DNN...")
        dnn_model = self.build_dnn_model(X_train_scaled.shape[1])
        dnn_model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        self.models['dnn'] = dnn_model
        
        # Train 1D-CNN
        print("Training 1D-CNN...")
        cnn_model = self.build_1d_cnn_model(X_train_scaled.shape[1])
        cnn_model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        self.models['cnn'] = cnn_model
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, y_test)
        
        self.is_trained = True
        print("All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\n=== Model Evaluation ===")
        
        for name, model in self.models.items():
            if name == 'xgboost':
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_proba = model.predict(X_test).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name.upper()} - Accuracy: {accuracy:.3f}")
    
    def load_or_train_models(self):
        """Load trained models or train new ones"""
        try:
            if all(os.path.exists(f'backend/models/{name}_model.joblib') for name in ['xgboost', 'scaler']):
                self.models['xgboost'] = joblib.load('backend/models/xgboost_model.joblib')
                self.scaler = joblib.load('backend/models/scaler.joblib')
                self.is_trained = True
                print("Pre-trained models loaded successfully!")
            else:
                print("Training new models...")
                self.train_models()
                # Save models
                os.makedirs('backend/models', exist_ok=True)
                joblib.dump(self.models['xgboost'], 'backend/models/xgboost_model.joblib')
                joblib.dump(self.scaler, 'backend/models/scaler.joblib')
        except Exception as e:
            print(f"Error loading models: {e}. Training new models...")
            self.train_models()
    
    def predict_risk(self, user_data):
        """Predict diabetic retinopathy risk using ensemble method"""
        try:
            # Preprocess user data
            features = self.preprocess_user_data(user_data)
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from all models
            predictions = []
            
            # XGBoost prediction
            xgb_prob = self.models['xgboost'].predict_proba(features_scaled)[0, 1]
            predictions.append(xgb_prob)
            
            # DNN prediction
            dnn_prob = self.models['dnn'].predict(features_scaled, verbose=0)[0, 0]
            predictions.append(dnn_prob)
            
            # CNN prediction
            cnn_prob = self.models['cnn'].predict(features_scaled, verbose=0)[0, 0]
            predictions.append(cnn_prob)
            
            # Ensemble average
            final_probability = np.mean(predictions)
            risk_score = int(final_probability * 100)
            
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
                'probability': round(final_probability, 3)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback prediction
            return {
                'risk_level': "Low Risk",
                'risk_score': 15,
                'probability': 0.15
            }
    
    def explain_prediction(self, user_data):
        """Provide explainable AI insights"""
        try:
            features = self.preprocess_user_data(user_data)[0]
            explanations = []
            
            # Age factor (索引0)
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
            
            # Diabetes duration (索引1)
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
            
            # HbA1c level (索引2)
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

            # Fasting Blood Glucose factor (索引3)
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
            elif fbg < 70:
                explanations.append({
                    'factor': 'Fasting Blood Glucose',
                    'impact': 'Medium',
                    'explanation': f'Fasting blood glucose of {fbg:.0f} mg/dL may be too low'
                })
             
            # Hypertension (索引4)
            if features[4] == 1.0:
                explanations.append({
                    'factor': 'High Blood Pressure',
                    'impact': 'Medium',
                    'explanation': 'Hypertension can accelerate retinopathy development'
                })
            
            # Nephropathy (索引5)
            if features[5] == 1.0:
                explanations.append({
                    'factor': 'Kidney Disease',
                    'impact': 'High',
                    'explanation': 'Diabetic kidney disease is closely linked to retinopathy'
                })
            
            # Neuropathy (索引6)
            if features[6] == 1.0:
                explanations.append({
                    'factor': 'Nerve Damage',
                    'impact': 'Medium',
                    'explanation': 'Diabetic neuropathy may indicate systemic complications'
                })
            
            # High cholesterol (索引7)
            if features[7] == 1.0:
                explanations.append({
                    'factor': 'High Cholesterol',
                    'impact': 'Low',
                    'explanation': 'High cholesterol can contribute to vascular complications'
                })
            
            # BMI (索引8)
            bmi = features[8]
            if bmi > 30:
                explanations.append({
                    'factor': 'Body Mass Index',
                    'impact': 'Medium',
                    'explanation': f'BMI of {bmi:.1f} indicates obesity, which can worsen diabetes control'
                })
            elif bmi > 25:
                explanations.append({
                    'factor': 'Body Mass Index',
                    'impact': 'Low',
                    'explanation': f'BMI of {bmi:.1f} indicates overweight status'
                })
            
            return explanations[:5]
            
        except Exception as e:
            print(f"Explanation error: {e}")
            return [{
                'factor': 'Basic Assessment',
                'impact': 'Low', 
                'explanation': 'Standard diabetes management recommended'
            }]

# For testing
if __name__ == "__main__":
    model = DRRiskModel()
    
    # Test prediction
    test_data = {
        'age': '55',
        'duration': '12', 
        'hba1c': 'Type 2 Diabetes',
        'fbg': '180',
        'hypertension': 'Yes',
        'nephropathy': 'No',
        'neuropathy': 'No',
        'cholesterol': 'Yes',
        'bmi': '32',
        'smoking': 'Never',
        'eye_exam': '3-5 years ago',
        'adherence': 'I sometimes miss a dose'
    }
    
    prediction = model.predict_risk(test_data)
    explanation = model.explain_prediction(test_data)
    
    print("Prediction:", prediction)
    print("Explanation:", explanation)