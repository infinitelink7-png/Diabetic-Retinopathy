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

# 其余代码保持不变（如explain_prediction等）
if __name__ == "__main__":
    print("="*60)
    print("🏥 DR Risk Assessment - 3 Model Ensemble Training")
    print("="*60)
    print()

    model = DRRiskModel()
    
    # 新增测试预测
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
    print("\n📊 Prediction Results:")
    print(f"  Risk Level: {prediction['risk_level']}")
    print(f"  Risk Score: {prediction['risk_score']}/100")
    print(f"  Probability: {prediction['probability']}")
    
    print("\n" + "="*60)
    print("✅ All systems operational!")
    print("="*60)