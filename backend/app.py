from flask import Flask, request, jsonify, send_from_directory 
from flask_cors import CORS
import json
import uuid
from datetime import datetime
import os 

app = Flask(__name__)
CORS(app)

# 配置
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///assessments.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 延迟导入以避免循环依赖
try:
    from database import init_db
    from models import RiskAssessment, db
    # 初始化数据库
    init_db(app)
    db_initialized = True
    print(" Database initialized successfully")
except Exception as e:
    db_initialized = False
    print(f" Database initialization failed: {e}")

# 初始化模型
try:
    from model_training import DRRiskModel
    model = DRRiskModel()
    model_loaded = True
    print(" AI model loaded successfully")
except Exception as e:
    model_loaded = False
    print(f" Failed to load AI model: {e}")

# 健康检查端点
@app.route('/api/health', methods=['GET'])
def health_check():
    status = {
        'status': 'healthy', 
        'message': 'API service is running normally',
        'database': db_initialized,
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

# 根路径
@app.route('/')
def serve_index():
    return send_from_directory('fronted', 'step1.html')

# 服务前端HTML文件
@app.route('/<path:filename>')
def serve_html_files(filename):
    if filename in ['step1.html', 'step2.html', 'step3.html', 'step4.html', 'step5.html', 'history.html']:
        return send_from_directory('fronted', filename)
    # 默认返回首页
    return send_from_directory('fronted', 'step1.html')

# 特定的HTML路由
@app.route('/home')
def serve_home():
    return send_from_directory('fronted', 'step1.html')

@app.route('/history')
def serve_history():
    return send_from_directory('fronted', 'history.html')

@app.route('/step1.html')
def redirect_step1():
    return send_from_directory('fronted', 'step1.html')

@app.route('/step2.html')
def redirect_step2():
    return send_from_directory('fronted', 'step2.html')

@app.route('/step3.html')
def redirect_step3():
    return send_from_directory('fronted', 'step3.html')

@app.route('/step4.html')
def redirect_step4():
    return send_from_directory('fronted', 'step4.html')

@app.route('/step5.html')
def redirect_step5():
    return send_from_directory('fronted', 'step5.html')

# 静态文件服务
@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('fronted', filename)

# 风险评估预测（保存到数据库）
@app.route('/api/predict', methods=['POST'])
def predict_risk():
    try:
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'AI model is not loaded. Please try again later.'
            }), 503

        if not db_initialized:
            return jsonify({
                'success': False,
                'error': 'Database is not initialized.'
            }), 503

        user_data = request.json
        
        # 生成会话ID（如果不存在）
        session_id = user_data.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # 预测风险
        prediction = model.predict_risk(user_data)
        explanation = model.explain_prediction(user_data)
        recommendations = generate_recommendations(prediction, explanation)
        
        # 保存到数据库
        assessment = RiskAssessment(
            session_id=session_id,
            input_data=json.dumps(user_data),
            risk_level=prediction['risk_level'],
            risk_score=prediction['risk_score'],
            probability=prediction['probability'],
            explanation=json.dumps(explanation),
            recommendations=json.dumps(recommendations)
        )
        
        db.session.add(assessment)
        db.session.commit()
        
        response = {
            'success': True,
            'prediction': prediction,
            'explanation': explanation,
            'recommendations': recommendations,
            'assessment_id': assessment.id,
            'session_id': session_id
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 根据会话ID获取评估历史
@app.route('/api/assessments/<session_id>', methods=['GET'])
def get_session_assessments(session_id):
    try:
        if not db_initialized:
            return jsonify({'success': False, 'error': 'Database not available'}), 503
            
        assessments = RiskAssessment.query.filter_by(session_id=session_id).order_by(RiskAssessment.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'assessments': [assessment.to_dict() for assessment in assessments],
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 获取所有评估记录（用于管理）
@app.route('/api/assessments', methods=['GET'])
def get_all_assessments():
    try:
        if not db_initialized:
            return jsonify({'success': False, 'error': 'Database not available'}), 503
            
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        assessments = RiskAssessment.query.order_by(
            RiskAssessment.created_at.desc()
        ).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        return jsonify({
            'success': True,
            'assessments': [assessment.to_dict() for assessment in assessments.items],
            'total_count': assessments.total,
            'total_pages': assessments.pages,
            'current_page': page
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 获取特定的评估记录
@app.route('/api/assessment/<assessment_id>', methods=['GET'])
def get_assessment(assessment_id):
    try:
        if not db_initialized:
            return jsonify({'success': False, 'error': 'Database not available'}), 503
            
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
@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        if not db_initialized:
            return jsonify({'success': False, 'error': 'Database not available'}), 503
            
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
    if explanation:
        for factor in explanation[:3]:  # Top 3 most important factors
            factor_name = factor['factor']
            
            if 'Blood Sugar' in factor_name or 'HbA1c' in factor_name or 'Fasting Blood Glucose' in factor_name:
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

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    import os
    from datetime import datetime

    # 获取 Render 提供的 PORT 环境变量（默认为5000）
    port = int(os.environ.get('PORT', 5000))

    print("🚀 Starting Diabetic Retinopathy Risk Assessment API")
    print(f"🌐 Host: 0.0.0.0")
    print(f"🔌 Port: {port}")
    print(f"📦 Database initialized: {db_initialized}")
    print(f"🤖 Model loaded: {model_loaded}")
    print(f"🕒 Started at: {datetime.now().isoformat()}")

    # Render 要求绑定 0.0.0.0
    app.run(host='0.0.0.0', port=port)
