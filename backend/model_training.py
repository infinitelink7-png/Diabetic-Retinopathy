# backend/model_training.py
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# 新增：从fusion_pipeline_full.py导入的常量和函数
RANDOM_STATE = 42
DATA_CSV = 'diabetic_retinopathy_survey.csv'  # 如果无CSV，注释掉并用合成数据
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def metrics(y_true, y_prob):
    """Calculate classification metrics"""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def ensemble_average(prob1, prob2, weight1=0.5):
    """Compute weighted average ensemble"""
    return weight1 * prob1 + (1 - weight1) * prob2
=======
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import os

# 修复 TensorFlow 导入
try:
    import tensorflow as tf
    keras = tf.keras
    layers = tf.keras.layers
    print(f"✅ TensorFlow {tf.__version__} Loading Successfully")
    print("✅ Keras Loading Successfully")
except ImportError as e:
    print(f"❌ TensorFlow Not installed: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    exit(1)
except Exception as e:
    print(f"❌ TensorFlow Import Error: {e}")
    exit(1)
>>>>>>> 3e95ae64aaccf64b1bd02e97a3536f7dc828c1f9

class DRRiskModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
<<<<<<< HEAD
        self.preprocessor = None  # 新增：预处理器
=======
>>>>>>> 3e95ae64aaccf64b1bd02e97a3536f7dc828c1f9
        self.feature_names = [
            'age', 'diabetes_duration', 'hba1c_level', 'fbg_level', 'has_hypertension',
            'has_nephropathy', 'has_neuropathy', 'has_high_cholesterol',
            'bmi', 'is_smoker', 'years_since_last_eye_exam', 'medication_adherence'
        ]
        self.is_trained = False
<<<<<<< HEAD
        self.optimal_weight = 0.5  # 默认权重，从val set优化
        self.load_or_train_models()
    
    def load_or_train_models(self):
        """加载或训练模型（整合fusion_pipeline_full.py逻辑，只用XGB × DNN）"""
        xgb_path = os.path.join(MODEL_DIR, 'xgb_model1.joblib')
        dnn_path = os.path.join(MODEL_DIR, 'dnn_model.keras')
        preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor.joblib')
        
        if os.path.exists(xgb_path) and os.path.exists(dnn_path) and os.path.exists(preprocessor_path):
            print("Loading pre-trained models...")
            self.models['xgboost'] = joblib.load(xgb_path)
            self.models['dnn'] = tf.keras.models.load_model(dnn_path)
            self.preprocessor = joblib.load(preprocessor_path)
            self.is_trained = True
            return
        
        print("Training models from scratch...")
        
        # 数据加载（用CSV或合成数据）
        if os.path.exists(DATA_CSV):
            df = pd.read_csv(DATA_CSV)
            if 'Retinopathy_prob' in df.columns:
                df = df.drop(columns=['Retinopathy_prob'])
            y = df['Retinopathy'].values
            X = df.drop(columns=['Retinopathy'])
        else:
            # 用原有合成数据
            df = self.create_training_data()
            y = df['has_diabetic_retinopathy'].values
            X = df.drop(columns=['has_diabetic_retinopathy'])
        
        # 识别列类型（从fusion_pipeline_full.py）
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # 构建预处理器
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
        
        # 数据拆分
        X_train_raw, X_test_raw, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_train_raw, y_train_full, test_size=0.15, stratify=y_train_full, random_state=RANDOM_STATE
        )
        
        # 拟合预处理器
        self.preprocessor.fit(X_train_raw)
        X_train = self.preprocessor.transform(X_train_raw)
        X_val = self.preprocessor.transform(X_val_raw)
        X_test = self.preprocessor.transform(X_test_raw)
        
        # 训练XGBoost (Model 1)
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
        )
        self.models['xgboost'].fit(X_train, y_train)
        
        # 训练DNN
        def build_dnn(input_dim):
            model = Sequential([
                Dense(256, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.4),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
            return model
        
        self.models['dnn'] = build_dnn(X_train.shape[1])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]
        self.models['dnn'].fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=200, batch_size=64, callbacks=callbacks, verbose=1
        )
        
        # 优化权重（从val set）
        xgb_prob_val = self.models['xgboost'].predict_proba(X_val)[:, 1]
        dnn_prob_val = self.models['dnn'].predict(X_val, verbose=0).ravel()
        best_auc = 0
        for w in np.arange(0, 1.05, 0.05):
            ensemble_prob = ensemble_average(xgb_prob_val, dnn_prob_val, w)
            auc = roc_auc_score(y_val, ensemble_prob)
            if auc > best_auc:
                best_auc = auc
                self.optimal_weight = w
        print(f"Optimal weight for XGB in ensemble: {self.optimal_weight:.2f}")
        
        # 保存模型和预处理器
        joblib.dump(self.models['xgboost'], os.path.join(MODEL_DIR, 'xgb_model1.joblib'))
        self.models['dnn'].save(os.path.join(MODEL_DIR, 'dnn_model.keras'))
        joblib.dump(self.preprocessor, os.path.join(MODEL_DIR, 'preprocessor.joblib'))
        
        self.is_trained = True
    
    def preprocess_user_data(self, user_data):
        """预处理用户数据（更新为用preprocessor）"""
        try:
            # 将user_data转换为DataFrame（匹配训练数据的列）
            df_user = pd.DataFrame([user_data])
            # 应用预处理器
            features = self.preprocessor.transform(df_user)
            return features
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return np.array([[50, 5, 6.0, 120.0, 0, 0, 0, 0, 25.0, 0, 5.0, 0.5]])  # 默认值
    
    def predict_risk(self, user_data):
        """预测风险（只用XGB × DNN ensemble）"""
        try:
            features = self.preprocess_user_data(user_data)
            
            # XGB预测
            xgb_prob = self.models['xgboost'].predict_proba(features)[:, 1][0]
            
            # DNN预测
            dnn_prob = self.models['dnn'].predict(features, verbose=0)[0, 0]
            
            # Ensemble平均（用优化权重）
            final_probability = ensemble_average(xgb_prob, dnn_prob, self.optimal_weight)
=======
        self.load_or_train_models()
    
    def preprocess_user_data(self, user_data):
        """Preprocess user data from frontend for model prediction"""
        try:
            features = np.zeros(len(self.feature_names))
            
            features[0] = float(user_data.get('age', 50))
            features[1] = float(user_data.get('duration', 5))
            
            hba1c_map = {'Normal': 5.0, 'Prediabetes': 6.0, 'Type 2 Diabetes': 7.5}
            hba1c_value = user_data.get('hba1c', 'Normal')
            features[2] = hba1c_map.get(hba1c_value, 6.0)

            fbg = user_data.get('fbg', '120')
            features[3] = float(fbg) if fbg else 120.0
            
            features[4] = 1.0 if user_data.get('hypertension') == 'Yes' else 0.0
            features[5] = 1.0 if user_data.get('nephropathy') == 'Yes' else 0.0
            features[6] = 1.0 if user_data.get('neuropathy') == 'Yes' else 0.0
            features[7] = 1.0 if user_data.get('cholesterol') == 'Yes' else 0.0
            
            bmi = user_data.get('bmi', 25.0)
            if isinstance(bmi, str):
                bmi = float(bmi) if bmi.replace('.', '').isdigit() else 25.0
            features[8] = float(bmi)
            
            smoking_status = user_data.get('smoking', 'Never')
            features[9] = 1.0 if smoking_status == 'Current' else 0.0
            
            eye_exam_map = {
                'Within the past year': 0.5,
                '1-2 years ago': 1.5,
                '3-5 years ago': 4.0,
                'Over 5 years ago': 7.0,
                'Never': 10.0
            }
            features[10] = eye_exam_map.get(user_data.get('eye_exam', 'Never'), 5.0)
            
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
            age = np.random.normal(58, 12)
            age = max(20, min(90, age))
            
            diabetes_duration = np.random.exponential(8)
            diabetes_duration = min(50, diabetes_duration)
            
            hba1c = np.random.normal(7.5, 1.5) if np.random.random() > 0.3 else np.random.normal(5.8, 0.5)
            hba1c = max(4.0, min(15.0, hba1c))

            fbg = np.random.normal(140, 30) if hba1c > 6.5 else np.random.normal(100, 15)
            fbg = max(70, min(300, fbg))
            
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
        print("📊 Generating training data...")
        df = self.create_training_data()
        
        X = df[self.feature_names]
        y = df['has_diabetic_retinopathy']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"DR prevalence: {y.mean():.2%}")
        
        # Train XGBoost
        print("\n🤖 Training XGBoost...")
        xgb_model = self.build_xgboost_model()
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgboost'] = xgb_model
        print("  ✅ XGBoost trained")
        
        # Train DNN
        print("🧠 Training DNN...")
        dnn_model = self.build_dnn_model(X_train_scaled.shape[1])
        dnn_model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        self.models['dnn'] = dnn_model
        print("  ✅ DNN trained")
        
        # Train 1D-CNN
        print("🔬 Training 1D-CNN...")
        cnn_model = self.build_1d_cnn_model(X_train_scaled.shape[1])
        cnn_model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        self.models['cnn'] = cnn_model
        print("  ✅ CNN trained")
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, y_test)
        
        self.is_trained = True
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\n=== Model Evaluation ===")
        
        for name, model in self.models.items():
            if name == 'xgboost':
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name.upper()} - Accuracy: {accuracy:.3f}")
    
    def load_or_train_models(self):
        """Load trained models or train new ones"""
        try:
            models_dir = 'backend/models'
            xgb_path = f'{models_dir}/xgboost_model.joblib'
            dnn_path = f'{models_dir}/dnn_model.h5'
            cnn_path = f'{models_dir}/cnn_model.h5'
            scaler_path = f'{models_dir}/scaler_model.joblib'
            
            all_models_exist = all(os.path.exists(p) for p in [xgb_path, dnn_path, cnn_path, scaler_path])
            
            if all_models_exist:
                print("📂 Loading pre-trained models...")
                
                self.models['xgboost'] = joblib.load(xgb_path)
                print("  ✅ XGBoost loaded")
                
                self.models['dnn'] = keras.models.load_model(dnn_path)
                print("  ✅ DNN loaded")
                
                self.models['cnn'] = keras.models.load_model(cnn_path)
                print("  ✅ CNN loaded")
                
                self.scaler = joblib.load(scaler_path)
                print("  ✅ Scaler loaded")
                
                self.is_trained = True
                print("✅ All pre-trained models loaded successfully!\n")
                
            else:
                print("🔄 No pre-trained models found. Training new models...\n")
                self.train_models()
                
                os.makedirs(models_dir, exist_ok=True)
                
                print("\n💾 Saving models...")
                joblib.dump(self.models['xgboost'], xgb_path)
                print("  ✅ XGBoost saved")
                
                self.models['dnn'].save(dnn_path)
                print("  ✅ DNN saved")
                
                self.models['cnn'].save(cnn_path)
                print("  ✅ CNN saved")
                
                joblib.dump(self.scaler, scaler_path)
                print("  ✅ Scaler saved")
                
                print("✅ All models trained and saved successfully!\n")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_risk(self, user_data):
        """Predict diabetic retinopathy risk using ensemble method"""
        try:
            features = self.preprocess_user_data(user_data)
            features_scaled = self.scaler.transform(features)
            
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
>>>>>>> 3e95ae64aaccf64b1bd02e97a3536f7dc828c1f9
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
<<<<<<< HEAD
=======
            
>>>>>>> 3e95ae64aaccf64b1bd02e97a3536f7dc828c1f9
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'risk_level': "Low Risk",
                'risk_score': 15,
                'probability': 0.15
            }
    
<<<<<<< HEAD
    # 保留原有方法：create_training_data, explain_prediction 等（如果需要）

# 其余代码保持不变（如explain_prediction等）
=======
    def explain_prediction(self, user_data):
        """Provide explainable AI insights"""
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
            elif fbg < 70:
                explanations.append({
                    'factor': 'Fasting Blood Glucose',
                    'impact': 'Medium',
                    'explanation': f'Fasting blood glucose of {fbg:.0f} mg/dL may be too low'
                })
             
            if features[4] == 1.0:
                explanations.append({
                    'factor': 'High Blood Pressure',
                    'impact': 'Medium',
                    'explanation': 'Hypertension can accelerate retinopathy development'
                })
            
            if features[5] == 1.0:
                explanations.append({
                    'factor': 'Kidney Disease',
                    'impact': 'High',
                    'explanation': 'Diabetic kidney disease is closely linked to retinopathy'
                })
            
            if features[6] == 1.0:
                explanations.append({
                    'factor': 'Nerve Damage',
                    'impact': 'Medium',
                    'explanation': 'Diabetic neuropathy may indicate systemic complications'
                })
            
            if features[7] == 1.0:
                explanations.append({
                    'factor': 'High Cholesterol',
                    'impact': 'Low',
                    'explanation': 'High cholesterol can contribute to vascular complications'
                })
            
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

>>>>>>> 3e95ae64aaccf64b1bd02e97a3536f7dc828c1f9
if __name__ == "__main__":
    print("="*60)
    print("🏥 DR Risk Assessment - 3 Model Ensemble Training")
    print("="*60)
    print()
<<<<<<< HEAD

    model = DRRiskModel()
    
    # 新增测试预测
    test_data = {
        'age': '55',
        'duration': '12',
=======
    
    model = DRRiskModel()
    
    # Test prediction
    test_data = {
        'age': '55',
        'duration': '12', 
>>>>>>> 3e95ae64aaccf64b1bd02e97a3536f7dc828c1f9
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
<<<<<<< HEAD
    prediction = model.predict_risk(test_data)
=======
    
    print("="*60)
    print("🧪 Testing prediction with sample data...")
    print("="*60)
    
    prediction = model.predict_risk(test_data)
    explanation = model.explain_prediction(test_data)
    
>>>>>>> 3e95ae64aaccf64b1bd02e97a3536f7dc828c1f9
    print("\n📊 Prediction Results:")
    print(f"  Risk Level: {prediction['risk_level']}")
    print(f"  Risk Score: {prediction['risk_score']}/100")
    print(f"  Probability: {prediction['probability']}")
    
<<<<<<< HEAD
=======
    print("\n💡 Risk Factors:")
    for exp in explanation:
        print(f"  • {exp['factor']} ({exp['impact']} impact): {exp['explanation']}")
    
>>>>>>> 3e95ae64aaccf64b1bd02e97a3536f7dc828c1f9
    print("\n" + "="*60)
    print("✅ All systems operational!")
    print("="*60)