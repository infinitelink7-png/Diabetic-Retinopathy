# backend/app.py
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
import traceback

app = Flask(__name__)

# 🗄️ 智能数据库配置
def get_database_uri():
    database_url = os.environ.get('DATABASE_URL', '')
    if database_url:
        if database_url.startswith('postgres://'):
            fixed_url = database_url.replace('postgres://', 'postgresql://', 1)
            print(f"📊 Using PostgreSQL: {fixed_url.split('@')[0]}...")
            return fixed_url
        else:
            print(f"📊 Using custom database: {database_url.split('@')[0]}...")
            return database_url
    else:
        try:
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
            print(f"📊 MySQL not available, using SQLite: {e}")
            return 'sqlite:///risk_assessment.db'

app.config['SQLALCHEMY_DATABASE_URI'] = get_database_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app,
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

# 初始化数据库
init_db(app)

# 延迟加载模型管理器
model_manager = None

def get_model_manager():
    global model_manager
    if model_manager is None:
        print("📄 Loading ML model manager...")
        model_manager = DRRiskModel()
        print("✅ Model manager loaded")
    return model_manager

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict_risk():
    # 支持浏览器 preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200

    try:
        user_data = request.get_json(force=True)
        if not isinstance(user_data, dict):
            raise ValueError("Invalid JSON payload")

        # ensure session id
        session_id = user_data.get('session_id') or str(uuid.uuid4())
        user_data['session_id'] = session_id

        # get model manager
        model = get_model_manager()

        # run prediction & explanation (wrap in try)
        try:
            prediction = model.predict_risk(user_data)
            explanation = model.explain_prediction(user_data)
        except Exception as e_inner:
            print("❌ Internal model error:", e_inner)
            traceback.print_exc()
            # fallback to safe default prediction
            prediction = {
                'risk_level': 'Unknown',
                'risk_score': 0,
                'probability': 0.0,
                'model_used': user_data.get('selected_model', 'ensemble')
            }
            explanation = [{'factor': 'ModelError', 'impact': 'Low', 'explanation': str(e_inner)}]

        # validate prediction structure and normalize types
        if not isinstance(prediction, dict):
            print("⚠️ prediction not dict, rebuilding safe prediction.")
            prediction = {
                'risk_level': 'Unknown',
                'risk_score': 0,
                'probability': float(prediction) if isinstance(prediction, (int, float)) else 0.0,
                'model_used': user_data.get('selected_model', 'ensemble')
            }

        # ensure keys exist with fallback
        risk_level = str(prediction.get('risk_level', 'Unknown'))
        risk_score = int(prediction.get('risk_score', 0))
        probability = float(prediction.get('probability', 0.0))
        model_used = str(prediction.get('model_used', user_data.get('selected_model', 'ensemble')))

        # generate recommendations safely
        try:
            recommendations = generate_recommendations(
                {'risk_level': risk_level, 'risk_score': risk_score, 'probability': probability},
                explanation if isinstance(explanation, list) else []
            )
        except Exception as e_rec:
            print("❌ Error generating recommendations:", e_rec)
            traceback.print_exc()
            recommendations = [{
                'type': 'routine',
                'title': 'Standard Care',
                'message': 'Please consult a healthcare professional for personalized advice.'
            }]

        # persist to database (use safe conversions)
        try:
            assessment = RiskAssessment(
                session_id=str(session_id),
                input_data=json.dumps(user_data),
                risk_level=str(risk_level),
                risk_score=float(risk_score),
                probability=float(probability),
                explanation=json.dumps(explanation),
                recommendations=json.dumps(recommendations),
                created_at=datetime.utcnow()
            )
            db.session.add(assessment)
            db.session.commit()
            assessment_id = str(assessment.id)
        except Exception as e_db:
            print("❌ DB save error:", e_db)
            traceback.print_exc()
            assessment_id = None

        # response payload
        response = {
            'success': True,
            'prediction': {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'probability': probability,
                'model_used': model_used
            },
            'explanation': explanation,
            'recommendations': recommendations,
            'assessment_id': assessment_id,
            'session_id': session_id
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error in predict_risk: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# The rest of your existing endpoints (assessments, stats, db-test, health, static serving)
@app.route('/api/assessments/<session_id>', methods=['GET', 'OPTIONS'])
def get_session_assessments(session_id):
    if request.method == 'OPTIONS':
        return '', 200
    try:
        assessments = RiskAssessment.query.filter_by(session_id=session_id).order_by(RiskAssessment.created_at.desc()).all()
        return jsonify({
            'success': True,
            'assessments': [a.to_dict() for a in assessments],
            'session_id': session_id
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/assessments', methods=['GET', 'OPTIONS'])
def get_all_assessments():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        assessments = RiskAssessment.query.order_by(RiskAssessment.created_at.desc()).all()
        return jsonify({
            'success': True,
            'assessments': [a.to_dict() for a in assessments],
            'total_count': len(assessments)
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/assessment/<assessment_id>', methods=['GET', 'OPTIONS'])
def get_assessment(assessment_id):
    if request.method == 'OPTIONS':
        return '', 200
    try:
        assessment = RiskAssessment.query.get(assessment_id)
        if not assessment:
            return jsonify({'success': False, 'error': 'Assessment not found'}), 404
        return jsonify({'success': True, 'assessment': assessment.to_dict()}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET', 'OPTIONS'])
def get_stats():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        total_assessments = RiskAssessment.query.count()
        risk_stats = {
            'high': RiskAssessment.query.filter_by(risk_level='High Risk').count(),
            'moderate': RiskAssessment.query.filter_by(risk_level='Moderate Risk').count(),
            'low': RiskAssessment.query.filter_by(risk_level='Low Risk').count()
        }
        return jsonify({'success': True, 'stats': {'total_assessments': total_assessments, 'risk_distribution': risk_stats}}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/db-status', methods=['GET'])
def db_status():
    try:
        count = RiskAssessment.query.count()
        db_url = app.config['SQLALCHEMY_DATABASE_URI']
        db_type = "Unknown"
        if 'mysql' in db_url:
            db_type = "MySQL (XAMPP)"
        elif 'postgresql' in db_url:
            db_type = "PostgreSQL (Render)"
        elif 'sqlite' in db_url:
            db_type = "SQLite"
        safe_db_url = re.sub(r':([^@]+)@', ':****@', db_url)
        return jsonify({'status': 'healthy', 'database_type': db_type, 'connection': safe_db_url, 'total_assessments': count, 'tables_working': True}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'database_type': 'Unknown', 'error': str(e), 'message': '数据库连接失败'}), 500

@app.route('/api/db-test', methods=['GET'])
def db_test():
    try:
        from models import RiskAssessment, db
        from datetime import datetime
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
        test_record = RiskAssessment.query.filter_by(session_id=test_assessment.session_id).first()
        if test_record:
            db.session.delete(test_record)
            db.session.commit()
        return jsonify({'success': True, 'message': '✅ 数据库读写测试成功！', 'test_performed': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': f'❌ 数据库测试失败: {str(e)}'}), 500

def generate_recommendations(prediction, explanation):
    recommendations = []
    # safe retrieval
    risk_level = prediction.get('risk_level', 'Low Risk') if isinstance(prediction, dict) else 'Low Risk'
    risk_score = float(prediction.get('risk_score', 0)) if isinstance(prediction, dict) else 0.0

    if risk_level == 'High Risk' or risk_score >= 70:
        recommendations.append({'type': 'urgent', 'title': 'Consult Doctor Immediately', 'message': 'We strongly recommend scheduling an appointment with an ophthalmologist for a comprehensive dilated eye examination within 1-3 months.'})
    elif risk_level == 'Moderate Risk' or risk_score >= 30:
        recommendations.append({'type': 'important', 'title': 'Regular Monitoring', 'message': 'We recommend an eye examination within 6 months and close monitoring of blood sugar and blood pressure.'})
    else:
        recommendations.append({'type': 'routine', 'title': 'Continue Screening', 'message': 'Please maintain good diabetes management and annual eye screening.'})

    # explanation may be list of factors
    try:
        for factor in (explanation or [])[:3]:
            factor_name = factor.get('factor', '') if isinstance(factor, dict) else str(factor)
            if 'Blood Sugar' in factor_name or 'HbA1c' in factor_name:
                recommendations.append({'type': 'management', 'title': 'Optimize Blood Sugar Control', 'message': 'Good blood sugar control is key to preventing diabetic retinopathy.'})
            elif 'Blood Pressure' in factor_name:
                recommendations.append({'type': 'management', 'title': 'Control Blood Pressure', 'message': 'High blood pressure can accelerate the development of diabetic retinopathy.'})
            elif 'Kidney' in factor_name or 'Nephropathy' in factor_name:
                recommendations.append({'type': 'specialist', 'title': 'Kidney Health', 'message': 'Diabetic kidney disease is closely related to retinopathy. We recommend kidney function tests.'})
            elif 'Diabetes Duration' in factor_name:
                recommendations.append({'type': 'monitoring', 'title': 'Enhanced Monitoring', 'message': 'Longer diabetes duration requires more frequent eye examinations.'})
    except Exception as e:
        print("⚠️ Error parsing explanation for recommendations:", e)

    return recommendations

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({'status': 'healthy', 'message': 'API service is running normally', 'cors_enabled': True, 'version': '1.0'}), 200

# static front-end serving (adjust path as your project uses)
@app.route('/fronted/<path:filename>')
def serve_fronted_files(filename):
    return send_from_directory('../fronted', filename)

@app.route('/home')
def serve_home():
    return send_from_directory('../fronted', 'step1.html')

@app.route('/')
def serve_index():
    return send_from_directory('../fronted', 'step1.html')

# map static pages
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
@app.route('/result.html')
def redirect_result():
    return send_from_directory('../fronted', 'result.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print(f"🚀 Starting server on 0.0.0.0:{port}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print(f"✅ Health Check: http://0.0.0.0:{port}/api/health")
    print(f"🔓 CORS: Fully enabled for all origins")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
