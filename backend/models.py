from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

# 初始化数据库对象
db = SQLAlchemy()

class RiskAssessment(db.Model):
    __tablename__ = 'risk_assessments'
    
    # 使用字符串 UUID 作为主键
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(100), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    probability = db.Column(db.Float, nullable=True)
    explanation = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,  # 直接返回字符串，不需要转换
            'session_id': self.session_id,
            'input_data': self.input_data,
            'risk_level': self.risk_level,
            'risk_score': self.risk_score,
            'probability': self.probability,
            'explanation': self.explanation,
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat()
        }