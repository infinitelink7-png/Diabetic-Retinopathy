# backend/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

class DRDataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        
    def create_preprocessor(self):
        # Numerical features
        numeric_features = ['age', 'diabetes_duration', 'bmi']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features
        categorical_features = ['diabetes_type', 'hypertension', 'nephropathy', 
                               'neuropathy', 'high_cholesterol', 'smoking_status',
                               'last_eye_exam', 'previous_dr_diagnosis', 'medication_adherence']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
    def fit(self, X):
        self.create_preprocessor()
        self.preprocessor.fit(X)
        return self
        
    def transform(self, X):
        return self.preprocessor.transform(X)
    
    def save(self, filepath):
        joblib.dump({
            'preprocessor': self.preprocessor
        }, filepath)
        
    def load(self, filepath):
        data = joblib.load(filepath)
        self.preprocessor = data['preprocessor']