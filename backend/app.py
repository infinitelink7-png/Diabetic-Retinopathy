from flask import Flask, request, jsonify, send_from_directory 
from flask_cors import CORS
from model_training import DRRiskModel
from database import init_db
from models import RiskAssessment, db
import json
import uuid
from datetime import datetime
import os 

app = Flask(__name__)
CORS(app)

# 初始化数据库
init_db(app)

# 初始化模型
model = DRRiskModel()

# 你的所有现有API路由保持不变...
@app.route('/api/predict', methods=['POST'])
def predict_risk():
    # ... 保持原有代码不变

@app.route('/api/assessments/<session_id>', methods=['GET'])
def get_session_assessments(session_id):
    # ... 保持原有代码不变

# ... 其他所有现有API路由都保持不变

# ↓↓↓ 只是在这些现有路由后面添加前端服务路由 ↓↓↓

# 服务前端HTML文件
@app.route('/frontend/<path:filename>')
def serve_frontend_files(filename):
    return send_from_directory('../fronted', filename)

# 服务主页面
@app.route('/home')
def serve_home():
    return send_from_directory('../fronted', 'step1.html')

# 服务历史页面  
@app.route('/history')
def serve_history():
    return send_from_directory('../fronted', 'history.html')

# 确保根路径也指向前端（修改原来的index路由）
@app.route('/')
def serve_index():
    return send_from_directory('../fronted', 'step1.html')

# 你的其他现有代码保持不变...
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)