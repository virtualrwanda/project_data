# app.py - Integrated with Deep Learning Model
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
import re
import joblib
import warnings

warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    from tensorflow import keras
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("Warning: TensorFlow not installed. Deep Learning features disabled.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///transformer_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size
ALLOWED_EXTENSIONS = {'csv', 'gz'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
CORS(app)
db = SQLAlchemy(app)
from flask_migrate import Migrate

migrate = Migrate(app, db)
# Deep Learning Model Configuration
DL_MODEL_DIR = 'dl_model'
DL_MODEL_PATH = os.path.join(DL_MODEL_DIR, 'transformer_fault_prediction_model_dl.keras')
DL_SCALER_PATH = os.path.join(DL_MODEL_DIR, 'scaler_dl.joblib')
DL_LABEL_MAPPING_PATH = os.path.join(DL_MODEL_DIR, 'fault_label_mapping_dl.joblib')

# Load Deep Learning models if available
dl_model = None
dl_scaler = None
dl_label_mapping = None

if DL_AVAILABLE and os.path.exists(DL_MODEL_PATH):
    try:
        dl_model = keras.models.load_model(DL_MODEL_PATH)
        dl_scaler = joblib.load(DL_SCALER_PATH)
        dl_label_mapping = joblib.load(DL_LABEL_MAPPING_PATH)
        print("✅ Deep Learning model loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading Deep Learning model: {e}")

# Database Models
# In your app.py, update the TransformerReading model:

# In your app.py, update the TransformerReading model:

class TransformerReading(db.Model):
    """Transformer readings model"""
    __tablename__ = 'transformer_readings'
    
    id = db.Column(db.Integer, primary_key=True)
    serial_number = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(50), nullable=False)
    reading_datetime = db.Column(db.DateTime, nullable=False)
    vibration = db.Column(db.Float)
    temperature = db.Column(db.Float)
    voltage = db.Column(db.Float)
    current = db.Column(db.Float)
    load_status = db.Column(db.String(20))
    fault = db.Column(db.String(50))
    year = db.Column(db.Integer)
    month = db.Column(db.Integer)
    day = db.Column(db.Integer)
    hour = db.Column(db.Integer)
    minute = db.Column(db.Integer)
    second = db.Column(db.Integer)
    day_of_week = db.Column(db.Integer)
    is_weekend = db.Column(db.Boolean)
    quarter = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # New ML columns - these will be added by Alembic or manually
    dl_prediction = db.Column(db.String(50), nullable=True)
    dl_confidence = db.Column(db.Float, nullable=True)
    rf_prediction = db.Column(db.String(50), nullable=True)
    rf_confidence = db.Column(db.Float, nullable=True)
    ensemble_prediction = db.Column(db.String(50), nullable=True)
    ensemble_confidence = db.Column(db.Float, nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'serial_number': self.serial_number,
            'location': self.location,
            'reading_datetime': self.reading_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'vibration': self.vibration,
            'temperature': self.temperature,
            'voltage': self.voltage,
            'current': self.current,
            'load_status': self.load_status,
            'fault': self.fault,
            'dl_prediction': self.dl_prediction,
            'dl_confidence': self.dl_confidence,
            'rf_prediction': self.rf_prediction,
            'rf_confidence': self.rf_confidence,
            'ensemble_prediction': self.ensemble_prediction,
            'ensemble_confidence': self.ensemble_confidence,
            'year': self.year,
            'month': self.month,
            'day': self.day,
            'hour': self.hour,
            'minute': self.minute,
            'second': self.second,
            'day_of_week': self.day_of_week,
            'is_weekend': self.is_weekend,
            'quarter': self.quarter
        }
class DailySummary(db.Model):
    """Daily summary model"""
    __tablename__ = 'transformer_daily_summary'
    
    id = db.Column(db.Integer, primary_key=True)
    serial_number = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(50), nullable=False)
    date = db.Column(db.Date, nullable=False)
    avg_current = db.Column(db.Float)
    max_current = db.Column(db.Float)
    min_current = db.Column(db.Float)
    avg_temperature = db.Column(db.Float)
    max_temperature = db.Column(db.Float)
    avg_vibration = db.Column(db.Float)
    max_vibration = db.Column(db.Float)
    fault_count = db.Column(db.Integer, default=0)
    over_load_minutes = db.Column(db.Float, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'serial_number': self.serial_number,
            'location': self.location,
            'date': self.date.strftime('%Y-%m-%d'),
            'avg_current': self.avg_current,
            'max_current': self.max_current,
            'min_current': self.min_current,
            'avg_temperature': self.avg_temperature,
            'max_temperature': self.max_temperature,
            'avg_vibration': self.avg_vibration,
            'max_vibration': self.max_vibration,
            'fault_count': self.fault_count,
            'over_load_minutes': self.over_load_minutes
        }

# Constants
LOCATIONS = ['Kigali', 'Bugesera', 'Rulindo', 'Gicumbi']
SERIAL_NUMBERS = {
    'Kigali': 'TRF-2024-KGL-001',
    'Bugesera': 'TRF-2024-BGS-001',
    'Rulindo': 'TRF-2024-RLD-001',
    'Gicumbi': 'TRF-2024-GCM-001'
}

FAULT_TYPES = [
    'Normal', 'Overheat', 'Mechanical looseness', 'Short circuit',
    'Overload + Thermal', 'Partial discharge', 'Voltage sag',
    'Harmonic distortion', 'Insulation failure'
]

# Map to DL model fault types
DL_FAULT_MAPPING = {
    'Normal': 'Normal',
    'Overheat': 'Overheating',
    'Mechanical looseness': 'Winding Fault',
    'Short circuit': 'Winding Fault',
    'Overload + Thermal': 'Overheating',
    'Partial discharge': 'Partial Discharge',
    'Voltage sag': 'Core Fault',
    'Harmonic distortion': 'Insulation Degradation',
    'Insulation failure': 'Insulation Degradation'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_datetime_with_seconds(dt_str):
    """Parse datetime string that might have seconds"""
    try:
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    except:
        try:
            return datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
        except:
            return datetime.fromisoformat(dt_str)

def predict_with_dl(vibration, temperature, voltage, current, load_status='normal'):
    """Make prediction using Deep Learning model"""
    if dl_model is None or dl_scaler is None:
        return None, None
    
    try:
        # Prepare features for DL model
        features_df = pd.DataFrame([{
            'vibration': vibration,
            'temperature': temperature,
            'voltage': voltage,
            'current': current,
            'load_status': load_status
        }])
        
        # Feature engineering
        features_df['reading_datetime'] = datetime.now()
        features_df['hour'] = features_df['reading_datetime'].dt.hour
        features_df['day_of_week'] = features_df['reading_datetime'].dt.dayofweek
        features_df['month'] = features_df['reading_datetime'].dt.month
        features_df['day_of_year'] = features_df['reading_datetime'].dt.dayofyear
        features_df['vibration_temperature_interaction'] = vibration * temperature
        features_df['vibration_squared'] = vibration ** 2
        features_df['temperature_squared'] = temperature ** 2
        
        # Handle load_status
        features_df['load_status_over'] = (load_status == 'over').astype(int)
        features_df['load_status_under'] = (load_status == 'under').astype(int)
        
        # Drop temporary columns
        features_df = features_df.drop(columns=['reading_datetime', 'load_status'])
        
        # Select only numeric columns for scaling
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_scaled = dl_scaler.transform(features_df[numeric_cols])
        
        # Make prediction
        prediction_proba = dl_model.predict(features_scaled, verbose=0)[0]
        
        # Get predicted class
        if len(prediction_proba) == 1:  # Binary classification
            pred_encoded = 1 if prediction_proba[0] > 0.5 else 0
            confidence = float(prediction_proba[0] * 100)
        else:  # Multi-class
            pred_encoded = np.argmax(prediction_proba)
            confidence = float(prediction_proba[pred_encoded] * 100)
        
        # Map to fault label
        if dl_label_mapping:
            if isinstance(dl_label_mapping, dict):
                pred_label = dl_label_mapping.get(pred_encoded, "Normal")
            else:
                pred_label = dl_label_mapping[pred_encoded] if pred_encoded < len(dl_label_mapping) else "Normal"
        else:
            # Default mapping if no label mapping available
            fault_map = {0: "Normal", 1: "Overheating", 2: "Winding Fault", 
                        3: "Insulation Degradation", 4: "Core Fault", 5: "Partial Discharge"}
            pred_label = fault_map.get(pred_encoded, "Normal")
        
        return pred_label, confidence
        
    except Exception as e:
        print(f"DL Prediction error: {e}")
        return None, None

def predict_simple_rf(vibration, temperature, voltage, current):
    """Simple rule-based prediction (fallback if no ML model)"""
    # Simple rules for fault detection
    if temperature > 90:
        return "Overheat", min(95, (temperature - 90) * 2)
    elif temperature > 75:
        return "Overheat", 70
    elif current > 800:
        return "Overload + Thermal", 85
    elif vibration > 5:
        return "Mechanical looseness", 80
    elif voltage < 22000 or voltage > 26000:
        return "Voltage sag", 75
    elif current > 600 and temperature > 70:
        return "Overload + Thermal", 70
    else:
        return "Normal", 90

def process_with_ml_prediction(reading):
    """Process a reading with ML models"""
    # Get values
    vibration = reading.vibration if reading.vibration else 0
    temperature = reading.temperature if reading.temperature else 0
    voltage = reading.voltage if reading.voltage else 0
    current = reading.current if reading.current else 0
    load_status = reading.load_status if reading.load_status else 'normal'
    
    # DL Prediction
    if dl_model is not None:
        dl_pred, dl_conf = predict_with_dl(vibration, temperature, voltage, current, load_status)
        reading.dl_prediction = dl_pred
        reading.dl_confidence = dl_conf
    
    # Simple RF prediction
    rf_pred, rf_conf = predict_simple_rf(vibration, temperature, voltage, current)
    reading.rf_prediction = rf_pred
    reading.rf_confidence = rf_conf
    
    # Ensemble prediction (weighted average)
    if dl_model is not None and dl_pred:
        # Convert to common fault types
        dl_fault = DL_FAULT_MAPPING.get(dl_pred, dl_pred)
        rf_fault = DL_FAULT_MAPPING.get(rf_pred, rf_pred)
        
        if dl_fault == rf_fault:
            reading.ensemble_prediction = dl_fault
            reading.ensemble_confidence = (dl_conf + rf_conf) / 2
        else:
            # Use the one with higher confidence
            if dl_conf > rf_conf:
                reading.ensemble_prediction = dl_fault
                reading.ensemble_confidence = dl_conf
            else:
                reading.ensemble_prediction = rf_fault
                reading.ensemble_confidence = rf_conf
    else:
        reading.ensemble_prediction = DL_FAULT_MAPPING.get(rf_pred, rf_pred)
        reading.ensemble_confidence = rf_conf
    
    return reading

def validate_csv_format(df):
    """Validate CSV format and return issues"""
    issues = []
    
    required_cols = ['serial_number', 'location', 'reading_datetime', 
                    'vibration', 'temperature', 'voltage', 'current']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {', '.join(missing)}")
    
    return issues

def process_csv_file(filepath):
    """Process and insert CSV data into database with ML predictions"""
    try:
        print(f"Processing file: {filepath}")
        
        # Read CSV file
        if filepath.endswith('.gz'):
            df = pd.read_csv(filepath, compression='gzip')
        else:
            df = pd.read_csv(filepath)
        
        print(f"Loaded {len(df)} rows from CSV")
        
        # Validate format
        issues = validate_csv_format(df)
        if issues:
            return False, f"CSV format issues: {'; '.join(issues)}"
        
        # Convert datetime
        df['reading_datetime'] = pd.to_datetime(df['reading_datetime'])
        
        # Handle boolean is_weekend column
        if 'is_weekend' in df.columns:
            df['is_weekend'] = df['is_weekend'].astype(str).str.lower().map({
                'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False
            }).fillna(False)
        else:
            df['is_weekend'] = df['reading_datetime'].dt.dayofweek >= 5
        
        # Ensure all time-based columns exist
        time_columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 'day_of_week', 'quarter']
        for col in time_columns:
            if col not in df.columns:
                if col == 'quarter':
                    df[col] = df['reading_datetime'].dt.quarter
                elif col == 'day_of_week':
                    df[col] = df['reading_datetime'].dt.dayofweek
                elif col == 'second':
                    df[col] = df['reading_datetime'].dt.second
                else:
                    df[col] = getattr(df['reading_datetime'].dt, col)
        
        # Ensure load_status and fault columns exist
        if 'load_status' not in df.columns:
            df['load_status'] = 'normal'
        if 'fault' not in df.columns:
            df['fault'] = 'Normal'
        
        # Clean data types
        df['serial_number'] = df['serial_number'].astype(str)
        df['location'] = df['location'].astype(str)
        df['vibration'] = pd.to_numeric(df['vibration'], errors='coerce').fillna(0)
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce').fillna(0)
        df['voltage'] = pd.to_numeric(df['voltage'], errors='coerce').fillna(0)
        df['current'] = pd.to_numeric(df['current'], errors='coerce').fillna(0)
        df['load_status'] = df['load_status'].astype(str).str.lower()
        df['fault'] = df['fault'].astype(str)
        
        # Remove rows with critical NaN values
        critical_cols = ['serial_number', 'location', 'reading_datetime']
        df = df.dropna(subset=critical_cols)
        
        if len(df) == 0:
            return False, "No valid rows found after data cleaning"
        
        print(f"After cleaning: {len(df)} valid rows")
        
        # Remove duplicates
        df['datetime_str'] = df['reading_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df = df.drop_duplicates(subset=['serial_number', 'datetime_str'], keep='first')
        
        # Insert data in batches with ML predictions
        batch_size = 5000
        total_inserted = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            records = []
            
            for _, row in batch.iterrows():
                try:
                    # Create reading object
                    reading = TransformerReading(
                        serial_number=row['serial_number'],
                        location=row['location'],
                        reading_datetime=row['reading_datetime'],
                        vibration=float(row['vibration']),
                        temperature=float(row['temperature']),
                        voltage=float(row['voltage']),
                        current=float(row['current']),
                        load_status=str(row['load_status']),
                        fault=str(row['fault']),
                        year=int(row['year']),
                        month=int(row['month']),
                        day=int(row['day']),
                        hour=int(row['hour']),
                        minute=int(row['minute']),
                        second=int(row['second']),
                        day_of_week=int(row['day_of_week']),
                        is_weekend=bool(row['is_weekend']),
                        quarter=int(row['quarter'])
                    )
                    
                    # Apply ML predictions
                    reading = process_with_ml_prediction(reading)
                    records.append(reading)
                    
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            if records:
                db.session.bulk_save_objects(records)
                db.session.commit()
                total_inserted += len(records)
                print(f"Inserted batch {i//batch_size + 1}: {len(records)} records")
        
        if total_inserted > 0:
            update_summary_tables()
            return True, f"Successfully inserted {total_inserted:,} records with ML predictions"
        else:
            return False, "No records were inserted"
        
    except Exception as e:
        db.session.rollback()
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Error processing file: {str(e)}"

def update_summary_tables():
    """Update daily and monthly summary tables"""
    try:
        print("Updating summary tables...")
        
        db.session.query(DailySummary).delete()
        
        readings = TransformerReading.query.all()
        
        if not readings:
            print("No readings found to summarize")
            return
        
        from collections import defaultdict
        daily_data = defaultdict(lambda: {
            'currents': [], 'temps': [], 'vibs': [],
            'fault_count': 0, 'overload_minutes': 0
        })
        
        for reading in readings:
            key = (reading.serial_number, reading.location, reading.reading_datetime.date())
            data = daily_data[key]
            data['currents'].append(reading.current)
            data['temps'].append(reading.temperature)
            data['vibs'].append(reading.vibration)
            if reading.fault != 'Normal':
                data['fault_count'] += 1
            if reading.load_status == 'over':
                data['overload_minutes'] += 20/60
        
        summaries = []
        for (serial, location, date), data in daily_data.items():
            summary = DailySummary(
                serial_number=serial,
                location=location,
                date=date,
                avg_current=np.mean(data['currents']) if data['currents'] else 0,
                max_current=np.max(data['currents']) if data['currents'] else 0,
                min_current=np.min(data['currents']) if data['currents'] else 0,
                avg_temperature=np.mean(data['temps']) if data['temps'] else 0,
                max_temperature=np.max(data['temps']) if data['temps'] else 0,
                avg_vibration=np.mean(data['vibs']) if data['vibs'] else 0,
                max_vibration=np.max(data['vibs']) if data['vibs'] else 0,
                fault_count=data['fault_count'],
                over_load_minutes=data['overload_minutes']
            )
            summaries.append(summary)
        
        if summaries:
            db.session.bulk_save_objects(summaries)
            db.session.commit()
            print(f"Created {len(summaries)} daily summaries")
        
    except Exception as e:
        db.session.rollback()
        print(f"Error updating summaries: {e}")

# Routes
@app.route('/')
def index():
    """Home page"""
    total_readings = TransformerReading.query.count()
    ml_status = "Active" if dl_model is not None else "Simple Rules Only"
    return render_template('index.html', locations=LOCATIONS, total_readings=total_readings, ml_status=ml_status)

@app.route('/upload', methods=['GET', 'POST'])
def upload_csv():
    """Upload and process CSV file"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            flash('Processing CSV file with ML predictions...', 'info')
            success, message = process_csv_file(filepath)
            
            if success:
                flash(message, 'success')
                try:
                    os.remove(filepath)
                except:
                    pass
                return redirect(url_for('readings'))
            else:
                flash(message, 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV or GZ file.', 'error')
            return redirect(request.url)
    
    total_records = TransformerReading.query.count()
    locations_count = db.session.query(TransformerReading.location).distinct().count()
    
    return render_template('upload.html', 
                         total_records=total_records,
                         locations_count=locations_count,
                         ml_available=dl_model is not None)

@app.route('/dashboard')
def dashboard():
    """Dashboard with statistics"""
    try:
        total_readings = TransformerReading.query.count()
        total_faults = TransformerReading.query.filter(
            TransformerReading.fault != 'Normal'
        ).count()
        
        # Get ML prediction accuracy comparison
        ml_accurate = 0
        if dl_model is not None:
            # Compare DL predictions with actual faults
            comparisons = TransformerReading.query.filter(
                TransformerReading.dl_prediction.isnot(None),
                TransformerReading.fault != 'Normal'
            ).all()
            if comparisons:
                correct = sum(1 for c in comparisons if c.dl_prediction == DL_FAULT_MAPPING.get(c.fault, c.fault))
                ml_accurate = (correct / len(comparisons)) * 100 if comparisons else 0
        
        faults_by_location = db.session.query(
            TransformerReading.location,
            db.func.count(TransformerReading.id).label('count')
        ).filter(
            TransformerReading.fault != 'Normal'
        ).group_by(TransformerReading.location).all()
        
        recent_faults = TransformerReading.query.filter(
            TransformerReading.fault != 'Normal'
        ).order_by(
            TransformerReading.reading_datetime.desc()
        ).limit(10).all()
        
        latest_readings = []
        for location in LOCATIONS:
            latest = TransformerReading.query.filter_by(
                location=location
            ).order_by(
                TransformerReading.reading_datetime.desc()
            ).first()
            if latest:
                latest_readings.append(latest)
        
        health_scores = []
        for reading in latest_readings:
            if reading.ensemble_confidence:
                health_scores.append(reading.ensemble_confidence)
            else:
                temp_score = max(0, 100 - (reading.temperature / 2))
                current_score = max(0, 100 - (reading.current / 20))
                vib_score = max(0, 100 - (reading.vibration * 10))
                health_score = (temp_score + current_score + vib_score) / 3
                health_scores.append(round(health_score, 1))
        
        stats = {
            'total_readings': total_readings,
            'total_faults': total_faults,
            'fault_rate': round((total_faults / total_readings * 100), 2) if total_readings > 0 else 0,
            'faults_by_location': faults_by_location,
            'recent_faults': recent_faults,
            'latest_readings': latest_readings,
            'health_scores': health_scores,
            'ml_available': dl_model is not None,
            'ml_accuracy': round(ml_accurate, 1) if ml_accurate else 0
        }
        
        return render_template('dashboard.html', stats=stats, locations=LOCATIONS)
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Make prediction using ML models"""
    if request.method == 'POST':
        try:
            # Get form data
            vibration = float(request.form.get('vibration', 0))
            temperature = float(request.form.get('temperature', 0))
            voltage = float(request.form.get('voltage', 0))
            current = float(request.form.get('current', 0))
            load_status = request.form.get('load_status', 'normal')
            
            # Make predictions
            dl_pred, dl_conf = predict_with_dl(vibration, temperature, voltage, current, load_status)
            rf_pred, rf_conf = predict_simple_rf(vibration, temperature, voltage, current)
            
            # Ensemble
            if dl_pred and dl_conf:
                dl_fault = DL_FAULT_MAPPING.get(dl_pred, dl_pred)
                rf_fault = DL_FAULT_MAPPING.get(rf_pred, rf_pred)
                
                if dl_fault == rf_fault:
                    ensemble_pred = dl_fault
                    ensemble_conf = (dl_conf + rf_conf) / 2
                else:
                    if dl_conf > rf_conf:
                        ensemble_pred = dl_fault
                        ensemble_conf = dl_conf
                    else:
                        ensemble_pred = rf_fault
                        ensemble_conf = rf_conf
            else:
                ensemble_pred = DL_FAULT_MAPPING.get(rf_pred, rf_pred)
                ensemble_conf = rf_conf
            
            prediction = {
                'dl': {'fault': dl_pred if dl_pred else 'N/A', 'confidence': dl_conf if dl_conf else 0},
                'rf': {'fault': rf_pred, 'confidence': rf_conf},
                'ensemble': {'fault': ensemble_pred, 'confidence': ensemble_conf},
                'values': {
                    'vibration': vibration,
                    'temperature': temperature,
                    'voltage': voltage,
                    'current': current,
                    'load_status': load_status
                }
            }
            
            return render_template('predict_result.html', prediction=prediction)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))
    
    return render_template('predict.html', ml_available=dl_model is not None)

@app.route('/readings')
def readings():
    """List all readings with pagination"""
    page = request.args.get('page', 1, type=int)
    location = request.args.get('location', '')
    fault_type = request.args.get('fault', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    query = TransformerReading.query
    
    if location and location != 'all':
        query = query.filter_by(location=location)
    
    if fault_type and fault_type != 'all':
        query = query.filter_by(fault=fault_type)
    
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            query = query.filter(TransformerReading.reading_datetime >= start)
        except:
            pass
    
    if end_date:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            end = end.replace(hour=23, minute=59, second=59)
            query = query.filter(TransformerReading.reading_datetime <= end)
        except:
            pass
    
    pagination = query.order_by(
        TransformerReading.reading_datetime.desc()
    ).paginate(page=page, per_page=50, error_out=False)
    
    fault_types = db.session.query(TransformerReading.fault).distinct().all()
    fault_types = [f[0] for f in fault_types]
    
    return render_template(
        'readings.html',
        readings=pagination.items,
        pagination=pagination,
        locations=LOCATIONS,
        fault_types=fault_types,
        current_location=location,
        current_fault=fault_type,
        start_date=start_date,
        end_date=end_date,
        ml_available=dl_model is not None
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API for prediction"""
    try:
        data = request.get_json()
        
        vibration = float(data.get('vibration', 0))
        temperature = float(data.get('temperature', 0))
        voltage = float(data.get('voltage', 0))
        current = float(data.get('current', 0))
        load_status = data.get('load_status', 'normal')
        
        dl_pred, dl_conf = predict_with_dl(vibration, temperature, voltage, current, load_status)
        rf_pred, rf_conf = predict_simple_rf(vibration, temperature, voltage, current)
        
        if dl_pred and dl_conf:
            dl_fault = DL_FAULT_MAPPING.get(dl_pred, dl_pred)
            rf_fault = DL_FAULT_MAPPING.get(rf_pred, rf_pred)
            
            if dl_fault == rf_fault:
                ensemble_pred = dl_fault
                ensemble_conf = (dl_conf + rf_conf) / 2
            else:
                if dl_conf > rf_conf:
                    ensemble_pred = dl_fault
                    ensemble_conf = dl_conf
                else:
                    ensemble_pred = rf_fault
                    ensemble_conf = rf_conf
        else:
            ensemble_pred = DL_FAULT_MAPPING.get(rf_pred, rf_pred)
            ensemble_conf = rf_conf
        
        return jsonify({
            'success': True,
            'prediction': {
                'dl': {'fault': dl_pred if dl_pred else 'N/A', 'confidence': dl_conf if dl_conf else 0},
                'rf': {'fault': rf_pred, 'confidence': rf_conf},
                'ensemble': {'fault': ensemble_pred, 'confidence': ensemble_conf}
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Existing routes (readings/add, edit, delete, etc.) remain the same
# [Keep all your existing routes from the original app.py]

@app.route('/readings/add', methods=['GET', 'POST'])
def add_reading():
    """Add a new reading"""
    if request.method == 'POST':
        try:
            reading = TransformerReading(
                serial_number=request.form['serial_number'],
                location=request.form['location'],
                reading_datetime=datetime.strptime(
                    request.form['reading_datetime'], 
                    '%Y-%m-%dT%H:%M'
                ),
                vibration=float(request.form['vibration']),
                temperature=float(request.form['temperature']),
                voltage=float(request.form['voltage']),
                current=float(request.form['current']),
                load_status=request.form['load_status'],
                fault=request.form['fault'],
                year=int(request.form['year']),
                month=int(request.form['month']),
                day=int(request.form['day']),
                hour=int(request.form['hour']),
                minute=int(request.form['minute']),
                second=int(request.form['second']),
                day_of_week=int(request.form['day_of_week']),
                is_weekend=request.form.get('is_weekend') == 'true',
                quarter=int(request.form['quarter'])
            )
            
            # Apply ML predictions
            reading = process_with_ml_prediction(reading)
            
            db.session.add(reading)
            db.session.commit()
            
            update_summary_tables()
            
            flash('Reading added successfully with ML predictions!', 'success')
            return redirect(url_for('readings'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding reading: {str(e)}', 'error')
    
    now = datetime.now()
    
    return render_template(
        'add_reading.html',
        locations=LOCATIONS,
        serial_numbers=SERIAL_NUMBERS,
        fault_types=FAULT_TYPES,
        now=now,
        ml_available=dl_model is not None
    )

# Keep all your other existing routes (edit_reading, delete_reading, view_reading, 
# analytics, faults, api endpoints, error handlers)

@app.errorhandler(404)
def not_found(error):
    return "404 - Page Not Found", 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return "500 - Internal Server Error", 500

def init_db():
    """Initialize database with tables"""
    with app.app_context():
        db.create_all()
        print("Database initialized")
        if dl_model is not None:
            print("Deep Learning model loaded and ready")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)