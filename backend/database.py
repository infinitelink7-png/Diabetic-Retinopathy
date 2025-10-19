import os
from flask_sqlalchemy import SQLAlchemy
from models import db

def init_db(app):
    # 🗄️ 使用 app.py 中的数据库配置，不再强制使用 SQLite
    # 这样就能继承 app.py 中的智能数据库选择逻辑
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
    print("Database initialized successfully!")
    print(f"Using database: {app.config['SQLALCHEMY_DATABASE_URI'].split('@')[0]}...")