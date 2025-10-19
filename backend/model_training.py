# backend/model_training.py
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
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

class DRRiskModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.preprocessor = None  # 新增：预处理器
        self.feature_names = [
            'age', 'diabetes_duration', 'hba1c_level', 'fbg_level', 'has_hypertension',
            'has_nephropathy', 'has_neuropathy', 'has_high_cholesterol',
            'bmi', 'is_smoker', 'years_since_last_eye_exam', 'medication_adherence'
        ]
        self.is_trained = False
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
            print("CSV columns:", df.columns.tolist())  # 打印 CSV 列名
            if 'Retinopathy_prob' in df.columns:
                df = df.drop(columns=['Retinopathy_prob'])
            y = df['Retinopathy'].values
            X = df.drop(columns=['Retinopathy'])
        else:
            # 用原有合成数据
            df = self.create_training_data()
            y = df['has_diabetic_retinopathy'].values
            X = df.drop(columns=['has_diabetic_retinopathy'])
        
        # 显式指定数值列和分类列
        num_cols = ['age', 'duration', 'hba1c', 'fbg', 'systolic_bp', 'diastolic_bp', 'Height_cm', 'Weight_kg', 'BMI']
        cat_cols = [
            'diabetes_type', 'Management_Insulin', 'Management_OralMed', 'Management_DietExercise',
            'blood_sugar_frequency', 'hypertension', 'nephropathy', 'neuropathy', 'cholesterol',
            'smoking', 'last_eye_exam', 'diagnosed_retinopathy', 'Vision_Blurriness',
            'Vision_Floaters', 'Vision_Fluctuating', 'Vision_Sudden_Loss', 'medication_adherence'
        ]
        
        # 确保所有列都在 X 中
        missing_cols = [col for col in num_cols + cat_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
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
        print(f"Training feature shape: {X_train.shape}")  # 打印预处理后维度
        X_val = self.preprocessor.transform(X_val_raw)
        X_test = self.preprocessor.transform(X_test_raw)
        
        # 训练XGBoost (Model 1)
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
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
    """
    预处理用户数据 - 将前端格式转换为CSV训练数据格式
    """
    try:
        print("\n" + "="*60)
        print("🔄 PREPROCESSING USER DATA")
        print("="*60)
        print(f"📥 Received data keys: {list(user_data.keys())}")
        
        # 创建与CSV列名完全匹配的数据字典
        processed_data = {}
        
        # ==================== 直接映射的字段 ====================
        processed_data['age'] = float(user_data.get('age', 50))
        processed_data['diabetes_type'] = user_data.get('diabetes_type', 'Type 2')
        processed_data['duration'] = float(user_data.get('duration', 5))
        processed_data['hba1c'] = float(user_data.get('hba1c', 7.0))
        processed_data['fbg'] = float(user_data.get('fbg', 120))
        
        # Management (已经是 0/1 格式)
        processed_data['Management_Insulin'] = int(user_data.get('Management_Insulin', 0))
        processed_data['Management_OralMed'] = int(user_data.get('Management_OralMed', 0))
        processed_data['Management_DietExercise'] = int(user_data.get('Management_DietExercise', 0))
        
        processed_data['blood_sugar_frequency'] = user_data.get('blood_sugar_frequency', 'Once a day')
        
        # Health conditions (已经是 0/1 格式)
        processed_data['hypertension'] = int(user_data.get('hypertension', 0))
        processed_data['systolic_bp'] = float(user_data.get('systolic_bp', 120))
        processed_data['diastolic_bp'] = float(user_data.get('diastolic_bp', 80))
        processed_data['nephropathy'] = int(user_data.get('nephropathy', 0))
        processed_data['neuropathy'] = int(user_data.get('neuropathy', 0))
        processed_data['cholesterol'] = int(user_data.get('cholesterol', 0))
        
        # Lifestyle
        processed_data['smoking'] = user_data.get('smoking', 'Never smoked')
        processed_data['Height_cm'] = float(user_data.get('Height_cm', 170))
        processed_data['Weight_kg'] = float(user_data.get('Weight_kg', 70))
        processed_data['BMI'] = float(user_data.get('BMI', 24.2))
        
        # Eye health
        processed_data['last_eye_exam'] = user_data.get('last_eye_exam', 'More than 2 years ago')
        processed_data['diagnosed_retinopathy'] = int(user_data.get('diagnosed_retinopathy', 0))
        processed_data['Vision_Blurriness'] = int(user_data.get('Vision_Blurriness', 0))
        processed_data['Vision_Floaters'] = int(user_data.get('Vision_Floaters', 0))
        processed_data['Vision_Fluctuating'] = int(user_data.get('Vision_Fluctuating', 0))
        processed_data['Vision_Sudden_Loss'] = int(user_data.get('Vision_Sudden_Loss', 0))
        processed_data['medication_adherence'] = user_data.get('medication_adherence', 'I never miss a dose')
        
        print(f"✅ Processed data sample:")
        print(f"   Age: {processed_data['age']}")
        print(f"   HbA1c: {processed_data['hba1c']}")
        print(f"   BMI: {processed_data['BMI']:.1f}")
        print(f"   Management_Insulin: {processed_data['Management_Insulin']}")
        
        # 转换为 DataFrame
        df_user = pd.DataFrame([processed_data])
        
        # 确保列顺序与训练时一致 (26列)
        expected_columns = [
            'age', 'diabetes_type', 'duration', 'hba1c', 'fbg',
            'Management_Insulin', 'Management_OralMed', 'Management_DietExercise',
            'blood_sugar_frequency', 'hypertension', 'systolic_bp', 'diastolic_bp',
            'nephropathy', 'neuropathy', 'cholesterol', 'smoking',
            'Height_cm', 'Weight_kg', 'BMI', 'last_eye_exam',
            'diagnosed_retinopathy', 'Vision_Blurriness', 'Vision_Floaters',
            'Vision_Fluctuating', 'Vision_Sudden_Loss', 'medication_adherence'
        ]
        
        # 检查缺失列
        missing_cols = [col for col in expected_columns if col not in df_user.columns]
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            for col in missing_cols:
                df_user[col] = 0 if col.startswith('Vision_') or col.startswith('Management_') else 'Unknown'
        
        # 按正确顺序排列
        df_user = df_user[expected_columns]
        
        print(f"✅ DataFrame shape: {df_user.shape}")
        print(f"📋 Columns: {df_user.columns.tolist()}")
        
        # 应用预处理器 (StandardScaler + OneHotEncoder)
        features = self.preprocessor.transform(df_user)
        
        print(f"✅ Final feature shape: {features.shape}")
        print("="*60 + "\n")
        
        return features
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回安全的默认特征
        print("⚠️ Using default fallback features")
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
        """预测风险（只用XGB × DNN ensemble）"""
        try:
            features = self.preprocess_user_data(user_data)
            # XGB预测
            xgb_prob = self.models['xgboost'].predict_proba(features)[:, 1][0]
            
            # DNN预测
            dnn_prob = self.models['dnn'].predict(features, verbose=0)[0, 0]
            
            # Ensemble平均（用优化权重）
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
            print(f"Prediction error: {e}")
            return {
                'risk_level': "Low Risk",
                'risk_score': 15,
                'probability': 0.15
            }
    
    # 保留原有方法：create_training_data, explain_prediction 等（如果需要）
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
    def explain_prediction(self, user_data):
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


# 其余代码保持不变（如explain_prediction等）
if __name__ == "__main__":
    model = DRRiskModel()
    test_data = {
        'age': 55,  # 数值，匹配 CSV 的 'age'
        'diabetes_type': 'Type 2',  # 匹配 CSV 的 'diabetes_type'
        'duration': 12,  # 数值，匹配 CSV 的 'duration'
        'hba1c': 7.5,  # 修正为数值，匹配 CSV 的 'hba1c'
        'fbg': 180,  # 数值，匹配 CSV 的 'fbg'
        'Management_Insulin': 0,  # 0 或 1，匹配 CSV
        'Management_OralMed': 1,  # 假设使用口服药
        'Management_DietExercise': 0,
        'blood_sugar_frequency': 'Once a day',  # 匹配 CSV 的值
        'hypertension': 1,  # 匹配 CSV
        'systolic_bp': 140,  # 当 hypertension='Yes' 时提供
        'diastolic_bp': 80,
        'nephropathy': 0,  # 匹配 CSV
        'neuropathy': 0,  # 匹配 CSV
        'cholesterol': 1,  # 匹配 CSV
        'smoking': 'Never smoked',  # 匹配 CSV
        'Height_cm': 170,  # 匹配 CSV
        'Weight_kg': 80,  # 匹配 CSV
        'BMI': 27.7,  # 计算为 Weight_kg / (Height_cm/100)^2
        'last_eye_exam': '1-2 years ago',  # 匹配 CSV
        'diagnosed_retinopathy': 0,  # 匹配 CSV
        'Vision_Blurriness': 0,  # 0 或 1，匹配 CSV
        'Vision_Floaters': 0,
        'Vision_Fluctuating': 0,
        'Vision_Sudden_Loss': 0,
        'medication_adherence': 'I occasionally miss doses'  # 匹配 CSV
    }
    prediction = model.predict_risk(test_data)
    print("\n📊 Prediction Results:")
    print(f"  Risk Level: {prediction['risk_level']}")
    print(f"  Risk Score: {prediction['risk_score']}/100")
    print(f"  Probability: {prediction['probability']}")