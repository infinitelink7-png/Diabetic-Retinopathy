from flask import Flask, request, jsonify, send_from_directory 
from flask_cors import CORS
import json
import uuid
from datetime import datetime
import os 

app = Flask(__name__)
CORS(app)

print("🚀 Starting Diabetic Retinopathy API for Render")

# 健康检查端点
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'message': 'API service is running normally on Render',
        'timestamp': datetime.now().isoformat()
    })

# 根路径
@app.route('/')
def serve_index():
    try:
        return send_from_directory('../fronted', 'step1.html')
    except Exception as e:
        return f"Frontend not found: {e}"

# HTML 页面路由
@app.route('/<path:filename>')
def serve_html_files(filename):
    if filename in ['step1.html', 'step2.html', 'step3.html', 'step4.html', 'step5.html', 'history.html']:
        try:
            return send_from_directory('../fronted', filename)
        except Exception as e:
            return f"File {filename} not found: {e}"
    return send_from_directory('../fronted', 'step1.html')

# 模拟预测端点
@app.route('/api/predict', methods=['POST'])
def predict_risk():
    try:
        user_data = request.json
        
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 基于输入数据的简单逻辑
        age = user_data.get('age', 50)
        diabetes_duration = user_data.get('diabetes_duration', 5)
        hba1c = user_data.get('hba1c', 7.0)
        blood_pressure_systolic = user_data.get('blood_pressure_systolic', 120)
        
        # 简单风险评估逻辑
        risk_score = min(100, max(0, 
            (hba1c - 6) * 10 + 
            (diabetes_duration - 1) * 3 +
            max(0, blood_pressure_systolic - 130) * 0.5 +
            (age - 40) * 0.2
        ))
        
        if risk_score >= 70:
            risk_level = 'High Risk'
        elif risk_score >= 30:
            risk_level = 'Moderate Risk'
        else:
            risk_level = 'Low Risk'
        
        prediction = {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 1),
            'probability': round(risk_score / 100, 2)
        }
        
        explanation = [
            {'factor': 'HbA1c Level', 'importance': 0.4, 'value': hba1c},
            {'factor': 'Diabetes Duration', 'importance': 0.3, 'value': diabetes_duration},
            {'factor': 'Blood Pressure', 'importance': 0.2, 'value': blood_pressure_systolic},
            {'factor': 'Age', 'importance': 0.1, 'value': age}
        ]
        
        recommendations = []
        if risk_level == 'High Risk':
            recommendations.append({
                'type': 'urgent',
                'title': 'Consult Doctor Immediately',
                'message': 'We strongly recommend scheduling an appointment with an ophthalmologist.',
                'action': 'Schedule Eye Doctor Appointment'
            })
        elif risk_level == 'Moderate Risk':
            recommendations.append({
                'type': 'important', 
                'title': 'Regular Monitoring',
                'message': 'We recommend an eye examination within 6 months.',
                'action': 'Schedule Eye Examination'
            })
        else:
            recommendations.append({
                'type': 'routine',
                'title': 'Continue Screening',
                'message': 'Please maintain good diabetes management.',
                'action': 'Continue Regular Screening'
            })
        
        response = {
            'success': True,
            'prediction': prediction,
            'explanation': explanation,
            'recommendations': recommendations,
            'session_id': session_id,
            'message': 'Assessment completed successfully'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # 关键：必须从环境变量获取端口
    port = int(os.environ.get('PORT', 10000))
    
    print(f"🚀 Starting Diabetic Retinopathy Risk Assessment API on Render")
    print(f"📍 PORT environment variable: {os.environ.get('PORT')}")
    print(f"📍 Using port: {port}")
    print(f"🌐 Host: 0.0.0.0")
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    
    # 关键：必须绑定到 0.0.0.0
    app.run(debug=False, host='0.0.0.0', port=port)