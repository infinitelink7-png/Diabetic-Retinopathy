"""Create and seed the SQLite database used by the app.

This script imports the SQLAlchemy `db` object and the `RiskAssessment` model
directly (not the whole Flask app) and creates `backend/assessments.db`.

Run this from the project root or the `backend` folder using the project's
virtualenv so dependencies are available:

Windows PowerShell:
    python backend/create_db_and_seed.py

This will create `backend/assessments.db` and insert 2 sample rows.
"""
from datetime import datetime
import os
import json

from models import db, RiskAssessment
from flask import Flask

def create_app_for_db():
    app = Flask(__name__)
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "assessments.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    db.init_app(app)
    return app

def seed_db(app):
    with app.app_context():
        db.create_all()

        # Check if any data exists
        if RiskAssessment.query.first():
            print('Database already contains data; skipping seeding.')
            return

        sample1 = RiskAssessment(
            session_id='demo-session-1',
            input_data=json.dumps({'age': 55, 'duration': 10}),
            risk_level='Moderate Risk',
            risk_score=45.0,
            probability=0.45,
            explanation=json.dumps([{'factor': 'Diabetes Duration', 'impact': 'Medium'}]),
            recommendations=json.dumps([{'title': 'Regular Monitoring'}]),
            created_at=datetime.utcnow()
        )

        sample2 = RiskAssessment(
            session_id='demo-session-2',
            input_data=json.dumps({'age': 70, 'duration': 20}),
            risk_level='High Risk',
            risk_score=85.0,
            probability=0.85,
            explanation=json.dumps([{'factor': 'Age', 'impact': 'High'}]),
            recommendations=json.dumps([{'title': 'Consult Doctor Immediately'}]),
            created_at=datetime.utcnow()
        )

        db.session.add_all([sample1, sample2])
        db.session.commit()
        print('Seeded 2 sample records into backend/assessments.db')

if __name__ == '__main__':
    app = create_app_for_db()
    seed_db(app)
