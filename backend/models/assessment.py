from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import uuid

db = SQLAlchemy()

class RiskAssessment(db.Model):
    __tablename__ = 'risk_assessments'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), nullable=False)  # 使用会话ID而不是用户ID
    
    # Input data
    input_data = db.Column(db.Text, nullable=False)  # JSON string
    
    # Prediction results
    risk_level = db.Column(db.String(20), nullable=False)
    risk_score = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    
    # Explanation and recommendations (JSON strings)
    explanation = db.Column(db.Text)
    recommendations = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'risk_level': self.risk_level,
            'risk_score': self.risk_score,
            'probability': self.probability,
            'explanation': json.loads(self.explanation) if self.explanation else [],
            'recommendations': json.loads(self.recommendations) if self.recommendations else [],
            'created_at': self.created_at.isoformat(),
            'input_data': json.loads(self.input_data)
        }