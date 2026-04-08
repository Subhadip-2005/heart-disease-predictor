from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle, numpy as np, jwt, datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heartapp.db'
db = SQLAlchemy(app)

# ── Models ────────────────────────────────────────────────────────────────────
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Prediction(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'))
    result     = db.Column(db.String(20))
    risk_pct   = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# ── Load model ────────────────────────────────────────────────────────────────
model  = pickle.load(open('RandomForestClassifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    user = User(username=data['username'],
                password=generate_password_hash(data['password']))
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'Registered successfully'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    token = jwt.encode({'user_id': user.id,
                        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)},
                       app.config['SECRET_KEY'])
    return jsonify({'token': token})

@app.route('/predict', methods=['POST'])
def predict():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    except:
        return jsonify({'error': 'Unauthorized'}), 401

    features = np.array([request.json['features']])
    scaled   = scaler.transform(features)
    pred     = model.predict(scaled)[0]
    prob     = model.predict_proba(scaled)[0][1] * 100

    p = Prediction(user_id=payload['user_id'],
                   result='High Risk' if pred == 1 else 'Low Risk',
                   risk_pct=round(prob, 1))
    db.session.add(p)
    db.session.commit()

    return jsonify({'result': p.result, 'risk_pct': p.risk_pct})

@app.route('/history', methods=['GET'])
def history():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    except:
        return jsonify({'error': 'Unauthorized'}), 401

    preds = Prediction.query.filter_by(user_id=payload['user_id'])\
                            .order_by(Prediction.created_at.desc()).all()
    return jsonify([{'result': p.result, 'risk': p.risk_pct,
                     'date': str(p.created_at)} for p in preds])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)