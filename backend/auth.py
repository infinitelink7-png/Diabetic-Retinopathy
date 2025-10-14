from flask import jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from models import User, db
from datetime import timedelta

jwt = JWTManager()

def init_auth(app):
    app.config['JWT_SECRET_KEY'] = 'your-jwt-secret-key'  # 在生产环境中应该使用环境变量
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
    
    jwt.init_app(app)

@jwt.user_identity_loader
def user_identity_lookup(user):
    return user.id

@jwt.user_lookup_loader
def user_lookup_callback(_jwt_header, jwt_data):
    identity = jwt_data["sub"]
    return User.query.get(identity)

def register_user(email, password, first_name=None, last_name=None):
    if User.query.filter_by(email=email).first():
        return None, "Email already registered"
    
    user = User(
        email=email,
        first_name=first_name,
        last_name=last_name
    )
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()
    
    return user, "User registered successfully"

def authenticate_user(email, password):
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=user)
        return access_token, user
    return None, None