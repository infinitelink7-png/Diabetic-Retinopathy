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
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///risk_assessment.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

# 初始化数据库
init_db(app)

# 初始化模型
# 延迟加载模型
model = None

def get_model():
    global model
    if model is None:
        print("🔄 Loading ML model...")
        model = DRRiskModel()
        print("✅ Model loaded successfully")
    return model


# 风险评估预测（保存到数据库）
@app.route('/api/predict', methods=['POST'])
def predict_risk():
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
        assessments = RiskAssessment.query.order_by(RiskAssessment.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'assessments': [assessment.to_dict() for assessment in assessments],
            'total_count': len(assessments)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 获取特定的评估记录
@app.route('/api/assessment/<assessment_id>', methods=['GET'])
def get_assessment(assessment_id):
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
@app.route('/api/stats', methods=['GET'])
def get_stats():
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
    for factor in explanation[:3]:  # Top 3 most important factors
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

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'API service is running normally'})

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
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
