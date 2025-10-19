from flask import Flask, request, jsonify, send_from_directory 
from flask_cors import CORS
from model_training import DRRiskModel
from database import init_db
from models import RiskAssessment, db
import json
import uuid
from datetime import datetime
import os 
import numpy as np
import re

app = Flask(__name__)

# 🗄️ 智能数据库配置
def get_database_uri():
    """
    智能选择数据库连接
    优先级: Render环境变量 > 本地XAMPP MySQL > SQLite
    """
    database_url = os.environ.get('DATABASE_URL', '')
    
    if database_url:
        # 生产环境 - Render PostgreSQL
        if database_url.startswith('postgres://'):
            # 修复PostgreSQL连接字符串
            fixed_url = database_url.replace('postgres://', 'postgresql://', 1)
            print(f"📊 Using PostgreSQL: {fixed_url.split('@')[0]}...")
            return fixed_url
        else:
            print(f"📊 Using custom database: {database_url.split('@')[0]}...")
            return database_url
    else:
        # 开发环境 - 尝试连接XAMPP MySQL
        try:
            # 测试MySQL连接
            import pymysql
            conn = pymysql.connect(
                host='localhost',
                user='root',
                password='',
                database='dr_risk_db',
                charset='utf8mb4'
            )
            conn.close()
            mysql_url = 'mysql+pymysql://root:@localhost/dr_risk_db?charset=utf8mb4'
            print("📊 Using XAMPP MySQL: mysql+pymysql://root:****@localhost/dr_risk_db")
            return mysql_url
        except Exception as e:
            # 回退到SQLite
            print(f"📊 MySQL not available, using SQLite: {e}")
            return 'sqlite:///risk_assessment.db'

app.config['SQLALCHEMY_DATABASE_URI'] = get_database_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 🔧 完全开放的 CORS 配置（用于调试）
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False)

# 添加全局 CORS 响应头
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

# 初始化数据库
init_db(app)

# 延迟加载模型
model = None

def get_model():
    global model
    if model is None:
        print("📄 Loading ML model...")
        model = DRRiskModel()  # 会自动训练或加载
        print("✅ Model loaded successfully")
    return model


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict_risk():
    # 明确处理 OPTIONS 预检请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response, 200
    
    try:
        user_data = request.json
        
        # 生成会话ID（如果不存在）
        session_id = user_data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())

        # 使用延迟加载的模型
        current_model = get_model()
        
        # 预测风险
        prediction = current_model.predict_risk(user_data)
        explanation = current_model.explain_prediction(user_data)
        recommendations = generate_recommendations(prediction, explanation)
        
        # 保存到数据库 - 转换为 Python 原生类型
        assessment = RiskAssessment(
            session_id=session_id,
            input_data=json.dumps(user_data),
            risk_level=str(prediction['risk_level']),
            risk_score=float(prediction['risk_score']),
            probability=float(prediction['probability']),
            explanation=json.dumps(explanation),
            recommendations=json.dumps(recommendations)
        )
        
        db.session.add(assessment)
        db.session.commit()
        
        # 返回数据 - 确保所有类型都是 JSON 可序列化的
        response = {
            'success': True,
            'prediction': {
                'risk_level': str(prediction['risk_level']),
                'risk_score': int(prediction['risk_score']),
                'probability': float(prediction['probability'])
            },
            'explanation': explanation,
            'recommendations': recommendations,
            'assessment_id': str(assessment.id),
            'session_id': str(session_id)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error in predict_risk: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 根据会话ID获取评估历史
@app.route('/api/assessments/<session_id>', methods=['GET', 'OPTIONS'])
def get_session_assessments(session_id):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        assessments = RiskAssessment.query.filter_by(session_id=session_id).order_by(RiskAssessment.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'assessments': [assessment.to_dict() for assessment in assessments],
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 获取所有评估记录（用于管理）
@app.route('/api/assessments', methods=['GET', 'OPTIONS'])
def get_all_assessments():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        assessments = RiskAssessment.query.order_by(RiskAssessment.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'assessments': [assessment.to_dict() for assessment in assessments],
            'total_count': len(assessments)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 获取特定的评估记录
@app.route('/api/assessment/<assessment_id>', methods=['GET', 'OPTIONS'])
def get_assessment(assessment_id):
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        assessment = RiskAssessment.query.get(assessment_id)
        
        if not assessment:
            return jsonify({'success': False, 'error': 'Assessment not found'}), 404
        
        return jsonify({
            'success': True,
            'assessment': assessment.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 获取统计信息
@app.route('/api/stats', methods=['GET', 'OPTIONS'])
def get_stats():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        total_assessments = RiskAssessment.query.count()
        
        # 风险等级统计
        risk_stats = {
            'high': RiskAssessment.query.filter_by(risk_level='High Risk').count(),
            'moderate': RiskAssessment.query.filter_by(risk_level='Moderate Risk').count(),
            'low': RiskAssessment.query.filter_by(risk_level='Low Risk').count()
        }
        
        return jsonify({
            'success': True,
            'stats': {
                'total_assessments': total_assessments,
                'risk_distribution': risk_stats
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db-status', methods=['GET'])
def db_status():
    """检查数据库连接状态"""
    try:
        from models import RiskAssessment
        # 测试数据库连接和基本操作
        count = RiskAssessment.query.count()
        
        db_url = app.config['SQLALCHEMY_DATABASE_URI']
        db_type = "Unknown"
        
        if 'mysql' in db_url:
            db_type = "MySQL (XAMPP)"
        elif 'postgresql' in db_url:
            db_type = "PostgreSQL (Render)"
        elif 'sqlite' in db_url:
            db_type = "SQLite"
            
        # 隐藏密码的安全显示
        safe_db_url = re.sub(r':([^@]+)@', ':****@', db_url)
        
        return jsonify({
            'status': 'healthy',
            'database_type': db_type,
            'connection': safe_db_url,
            'total_assessments': count,
            'tables_working': True
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'database_type': 'Unknown',
            'error': str(e),
            'message': '数据库连接失败'
        }), 500

@app.route('/api/db-test', methods=['GET'])
def db_test():
    """测试数据库读写操作"""
    try:
        from models import RiskAssessment, db
        from datetime import datetime
        
        # 创建测试记录
        test_assessment = RiskAssessment(
            session_id='test_session_' + str(uuid.uuid4())[:8],
            input_data='{"test": "data"}',
            risk_level='Low Risk',
            risk_score=10.0,
            probability=0.1,
            explanation='[]',
            recommendations='[]',
            created_at=datetime.utcnow()
        )
        
        db.session.add(test_assessment)
        db.session.commit()
        
        # 读取测试记录
        test_record = RiskAssessment.query.filter_by(session_id=test_assessment.session_id).first()
        
        # 清理测试记录
        if test_record:
            db.session.delete(test_record)
            db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '✅ 数据库读写测试成功！',
            'test_performed': True
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'❌ 数据库测试失败: {str(e)}'
        }), 500

def generate_recommendations(prediction, explanation):
    """Generate personalized recommendations"""
    recommendations = []
    risk_level = prediction['risk_level']
    risk_score = prediction['risk_score']
    
    # Recommendations based on risk level
    if risk_level == 'High Risk' or risk_score >= 70:
        recommendations.append({
            'type': 'urgent',
            'title': 'Consult Doctor Immediately',
            'message': 'We strongly recommend scheduling an appointment with an ophthalmologist for a comprehensive dilated eye examination within 1-3 months.',
            'action': 'Schedule Eye Doctor Appointment'
        })
    elif risk_level == 'Moderate Risk' or risk_score >= 30:
        recommendations.append({
            'type': 'important', 
            'title': 'Regular Monitoring',
            'message': 'We recommend an eye examination within 6 months and close monitoring of blood sugar and blood pressure.',
            'action': 'Schedule Eye Examination'
        })
    else:
        recommendations.append({
            'type': 'routine',
            'title': 'Continue Screening',
            'message': 'Please maintain good diabetes management and annual eye screening.',
            'action': 'Continue Regular Screening'
        })
    
    # Specific recommendations based on risk factors
    for factor in explanation[:3]:
        factor_name = factor['factor']
        
        if 'Blood Sugar' in factor_name or 'HbA1c' in factor_name:
            recommendations.append({
                'type': 'management',
                'title': 'Optimize Blood Sugar Control',
                'message': 'Good blood sugar control is key to preventing diabetic retinopathy.',
                'action': 'Consult Endocrinologist'
            })
        elif 'Blood Pressure' in factor_name:
            recommendations.append({
                'type': 'management',
                'title': 'Control Blood Pressure',
                'message': 'High blood pressure can accelerate the development of diabetic retinopathy.',
                'action': 'Monitor and Control Blood Pressure'
            })
        elif 'Kidney' in factor_name or 'Nephropathy' in factor_name:
            recommendations.append({
                'type': 'specialist',
                'title': 'Kidney Health',
                'message': 'Diabetic kidney disease is closely related to retinopathy. We recommend kidney function tests.',
                'action': 'Consult Nephrologist'
            })
        elif 'Diabetes Duration' in factor_name:
            recommendations.append({
                'type': 'monitoring',
                'title': 'Enhanced Monitoring',
                'message': 'Longer diabetes duration requires more frequent eye examinations.',
                'action': 'Increase Eye Check Frequency'
            })
    
    return recommendations

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 200
    
    return jsonify({
        'status': 'healthy', 
        'message': 'API service is running normally',
        'cors_enabled': True,
        'version': '1.0'
    }), 200

# 服务前端HTML文件
@app.route('/fronted/<path:filename>')
def serve_fronted_files(filename):
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

# 添加重定向路由来处理直接访问的HTML文件
@app.route('/step1.html')
def redirect_step1():
    return send_from_directory('../fronted', 'step1.html')

@app.route('/step2.html')
def redirect_step2():
    return send_from_directory('../fronted', 'step2.html')

@app.route('/step3.html')
def redirect_step3():
    return send_from_directory('../fronted', 'step3.html')

@app.route('/step4.html')
def redirect_step4():
    return send_from_directory('../fronted', 'step4.html')

@app.route('/step5.html')
def redirect_step5():
    return send_from_directory('../fronted', 'step5.html')

@app.route('/history.html')
def redirect_history():
    return send_from_directory('../fronted', 'history.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print(f"🚀 Starting server on 0.0.0.0:{port}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print(f"✅ Health Check: http://0.0.0.0:{port}/api/health")
    print(f"🔓 CORS: Fully enabled for all origins")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)