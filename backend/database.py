import os
from flask_sqlalchemy import SQLAlchemy
from models import db

def init_db(app):
    # 配置SQLite数据库
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "assessments.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
    print("Database initialized successfully!")