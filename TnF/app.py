# app.py - Updated with enhanced CSV upload handling
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

# Database Models
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

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_datetime_with_seconds(dt_str):
    """Parse datetime string that might have seconds"""
    try:
        # Try format with seconds first
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    except:
        try:
            # Try format without seconds
            return datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
        except:
            # Try ISO format
            return datetime.fromisoformat(dt_str)

def validate_csv_format(df):
    """Validate CSV format and return issues"""
    issues = []
    
    # Check required columns
    required_cols = ['serial_number', 'location', 'reading_datetime', 
                    'vibration', 'temperature', 'voltage', 'current']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {', '.join(missing)}")
    
    # Check data types
    if 'reading_datetime' in df.columns:
        try:
            pd.to_datetime(df['reading_datetime'])
        except:
            issues.append("reading_datetime column has invalid date format")
    
    # Check numeric columns
    numeric_cols = ['vibration', 'temperature', 'voltage', 'current']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.to_numeric(df[col], errors='coerce').notna().all():
                issues.append(f"{col} column contains non-numeric values")
    
    return issues

def process_csv_file(filepath):
    """Process and insert CSV data into database"""
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
        
        # Handle boolean is_weekend column (convert from string/object to boolean)
        if 'is_weekend' in df.columns:
            df['is_weekend'] = df['is_weekend'].astype(str).str.lower().map({
                'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False
            }).fillna(False)
        else:
            # Calculate if not present
            df['is_weekend'] = df['reading_datetime'].dt.dayofweek >= 5
        
        # Ensure all time-based columns exist
        if 'year' not in df.columns:
            df['year'] = df['reading_datetime'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['reading_datetime'].dt.month
        if 'day' not in df.columns:
            df['day'] = df['reading_datetime'].dt.day
        if 'hour' not in df.columns:
            df['hour'] = df['reading_datetime'].dt.hour
        if 'minute' not in df.columns:
            df['minute'] = df['reading_datetime'].dt.minute
        if 'second' not in df.columns:
            df['second'] = df['reading_datetime'].dt.second
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['reading_datetime'].dt.dayofweek
        if 'quarter' not in df.columns:
            df['quarter'] = df['reading_datetime'].dt.quarter
        
        # Ensure load_status and fault columns exist
        if 'load_status' not in df.columns:
            df['load_status'] = 'normal'
        if 'fault' not in df.columns:
            df['fault'] = 'Normal'
        
        # Clean data types
        df['serial_number'] = df['serial_number'].astype(str)
        df['location'] = df['location'].astype(str)
        df['vibration'] = pd.to_numeric(df['vibration'], errors='coerce')
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        df['voltage'] = pd.to_numeric(df['voltage'], errors='coerce')
        df['current'] = pd.to_numeric(df['current'], errors='coerce')
        df['load_status'] = df['load_status'].astype(str).str.lower()
        df['fault'] = df['fault'].astype(str)
        
        # Remove rows with critical NaN values
        critical_cols = ['serial_number', 'location', 'reading_datetime', 
                        'current', 'temperature', 'vibration']
        df = df.dropna(subset=critical_cols)
        
        if len(df) == 0:
            return False, "No valid rows found after data cleaning"
        
        print(f"After cleaning: {len(df)} valid rows")
        
        # Check for duplicates (same serial and datetime)
        df['datetime_str'] = df['reading_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        duplicates = df.duplicated(subset=['serial_number', 'datetime_str'], keep=False)
        if duplicates.any():
            print(f"Found {duplicates.sum()} duplicate records. Keeping first occurrence.")
            df = df.drop_duplicates(subset=['serial_number', 'datetime_str'], keep='first')
        
        # Insert data in batches
        batch_size = 10000
        total_inserted = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            records = []
            
            for _, row in batch.iterrows():
                try:
                    record = TransformerReading(
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
                    records.append(record)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            if records:
                db.session.bulk_save_objects(records)
                db.session.commit()
                total_inserted += len(records)
                print(f"Inserted batch {i//batch_size + 1}: {len(records)} records")
        
        if total_inserted > 0:
            # Update summary tables
            update_summary_tables()
            return True, f"Successfully inserted {total_inserted:,} records out of {len(df)} valid rows"
        else:
            return False, "No records were inserted. Please check your data format."
        
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
        
        # Clear existing summaries
        db.session.query(DailySummary).delete()
        
        # Get all readings
        readings = TransformerReading.query.all()
        
        if not readings:
            print("No readings found to summarize")
            return
        
        # Group by serial_number, location, and date
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
                data['overload_minutes'] += 20/60  # 20 seconds intervals
        
        # Create daily summaries
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
        
        # Bulk insert summaries
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
    return render_template('index.html', locations=LOCATIONS, total_readings=total_readings)

@app.route('/upload', methods=['GET', 'POST'])
def upload_csv():
    """Upload and process CSV file"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the CSV file
            flash('Processing CSV file... This may take a few moments.', 'info')
            success, message = process_csv_file(filepath)
            
            if success:
                flash(message, 'success')
                # Remove temp file after successful processing
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
    
    # Get existing data statistics
    total_records = TransformerReading.query.count()
    locations_count = db.session.query(TransformerReading.location).distinct().count()
    
    return render_template('upload.html', 
                         total_records=total_records,
                         locations_count=locations_count)

@app.route('/dashboard')
def dashboard():
    """Dashboard with statistics"""
    try:
        # Get statistics
        total_readings = TransformerReading.query.count()
        total_faults = TransformerReading.query.filter(
            TransformerReading.fault != 'Normal'
        ).count()
        
        # Faults by location
        faults_by_location = db.session.query(
            TransformerReading.location,
            db.func.count(TransformerReading.id).label('count')
        ).filter(
            TransformerReading.fault != 'Normal'
        ).group_by(TransformerReading.location).all()
        
        # Recent faults
        recent_faults = TransformerReading.query.filter(
            TransformerReading.fault != 'Normal'
        ).order_by(
            TransformerReading.reading_datetime.desc()
        ).limit(10).all()
        
        # Current status (latest readings)
        latest_readings = []
        for location in LOCATIONS:
            latest = TransformerReading.query.filter_by(
                location=location
            ).order_by(
                TransformerReading.reading_datetime.desc()
            ).first()
            if latest:
                latest_readings.append(latest)
        
        # Calculate health score for each transformer
        health_scores = []
        for reading in latest_readings:
            # Simple health score calculation
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
            'health_scores': health_scores
        }
        
        return render_template('dashboard.html', stats=stats, locations=LOCATIONS)
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/readings')
def readings():
    """List all readings with pagination"""
    page = request.args.get('page', 1, type=int)
    location = request.args.get('location', '')
    fault_type = request.args.get('fault', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    # Build query
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
    
    # Paginate
    pagination = query.order_by(
        TransformerReading.reading_datetime.desc()
    ).paginate(page=page, per_page=50, error_out=False)
    
    # Get unique fault types for filter
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
        end_date=end_date
    )

@app.route('/readings/add', methods=['GET', 'POST'])
def add_reading():
    """Add a new reading"""
    if request.method == 'POST':
        try:
            # Get form data
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
            
            db.session.add(reading)
            db.session.commit()
            
            # Update summary
            update_summary_tables()
            
            flash('Reading added successfully!', 'success')
            return redirect(url_for('readings'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding reading: {str(e)}', 'error')
    
    # Get current datetime for default value
    now = datetime.now()
    
    return render_template(
        'add_reading.html',
        locations=LOCATIONS,
        serial_numbers=SERIAL_NUMBERS,
        fault_types=FAULT_TYPES,
        now=now
    )

@app.route('/readings/<int:id>/edit', methods=['GET', 'POST'])
def edit_reading(id):
    """Edit a reading"""
    reading = TransformerReading.query.get_or_404(id)
    
    if request.method == 'POST':
        try:
            reading.serial_number = request.form['serial_number']
            reading.location = request.form['location']
            reading.reading_datetime = datetime.strptime(
                request.form['reading_datetime'], 
                '%Y-%m-%dT%H:%M'
            )
            reading.vibration = float(request.form['vibration'])
            reading.temperature = float(request.form['temperature'])
            reading.voltage = float(request.form['voltage'])
            reading.current = float(request.form['current'])
            reading.load_status = request.form['load_status']
            reading.fault = request.form['fault']
            reading.year = int(request.form['year'])
            reading.month = int(request.form['month'])
            reading.day = int(request.form['day'])
            reading.hour = int(request.form['hour'])
            reading.minute = int(request.form['minute'])
            reading.second = int(request.form['second'])
            reading.day_of_week = int(request.form['day_of_week'])
            reading.is_weekend = request.form.get('is_weekend') == 'true'
            reading.quarter = int(request.form['quarter'])
            
            db.session.commit()
            
            # Update summary
            update_summary_tables()
            
            flash('Reading updated successfully!', 'success')
            return redirect(url_for('readings'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating reading: {str(e)}', 'error')
    
    return render_template(
        'edit_reading.html',
        reading=reading,
        locations=LOCATIONS,
        serial_numbers=SERIAL_NUMBERS,
        fault_types=FAULT_TYPES
    )

@app.route('/readings/<int:id>/delete', methods=['POST'])
def delete_reading(id):
    """Delete a reading"""
    reading = TransformerReading.query.get_or_404(id)
    
    try:
        db.session.delete(reading)
        db.session.commit()
        
        # Update summary
        update_summary_tables()
        
        flash('Reading deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting reading: {str(e)}', 'error')
    
    return redirect(url_for('readings'))

@app.route('/readings/<int:id>')
def view_reading(id):
    """View a single reading"""
    reading = TransformerReading.query.get_or_404(id)
    return render_template('view_reading.html', reading=reading)

@app.route('/analytics')
def analytics():
    """Analytics and charts"""
    return render_template('analytics.html', locations=LOCATIONS)
# Add to app.py - Advanced Analytics Endpoints
# Add to app.py - Advanced Analytics Endpoints with Error Handling

@app.route('/api/advanced-analytics')
def advanced_analytics():
    """Advanced analytics API endpoint"""
    location = request.args.get('location', 'all')
    period = int(request.args.get('period', 30))
    analysis_type = request.args.get('analysis', 'trend')
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period)
        
        # Build query
        query = TransformerReading.query
        if location != 'all':
            query = query.filter_by(location=location)
        
        query = query.filter(
            TransformerReading.reading_datetime >= start_date,
            TransformerReading.reading_datetime <= end_date
        )
        
        readings = query.all()
        
        if not readings:
            return jsonify({'success': False, 'error': 'No data available for the selected period'})
        
        # Convert to DataFrame for analysis
        df_list = []
        for r in readings:
            df_list.append({
                'datetime': r.reading_datetime,
                'temperature': r.temperature,
                'current': r.current,
                'vibration': r.vibration,
                'voltage': r.voltage,
                'fault': r.fault,
                'load_status': r.load_status,
                'hour': r.hour,
                'month': r.month,
                'day_of_week': r.day_of_week
            })
        
        df = pd.DataFrame(df_list)
        
        if len(df) == 0:
            return jsonify({'success': False, 'error': 'No valid data points available'})
        
        # Calculate KPIs
        kpis = calculate_advanced_kpis(df)
        
        # Trend analysis
        trend = calculate_trend_analysis(df)
        
        # Correlation matrix
        correlation = calculate_correlation_matrix(df)
        
        # Anomaly detection
        anomalies = detect_anomalies(df)
        
        # Predictive forecast
        forecast = generate_forecast(df)
        
        # Fault distribution
        faults = get_fault_distribution(df)
        
        # Health index
        health = calculate_health_index_advanced(df)
        
        # Seasonal patterns
        seasonal = analyze_seasonal_patterns(df)
        
        # Hourly patterns
        hourly = analyze_hourly_patterns(df)
        
        # Risk assessment
        risk = assess_risks(df)
        
        # Generate insights
        insights = generate_advanced_insights(df, kpis, anomalies, health)
        
        return jsonify({
            'success': True,
            'kpis': kpis,
            'trend': trend,
            'correlation': correlation,
            'anomalies': anomalies,
            'forecast': forecast,
            'faults': faults,
            'health': health,
            'seasonal': seasonal,
            'hourly': hourly,
            'risk': risk,
            'insights': insights
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

def calculate_advanced_kpis(df):
    """Calculate advanced KPIs with error handling"""
    try:
        # Ensure numeric columns
        df['current'] = pd.to_numeric(df['current'], errors='coerce')
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        
        # Drop NaN values for calculations
        current_clean = df['current'].dropna()
        temp_clean = df['temperature'].dropna()
        
        avg_current = current_clean.mean() if len(current_clean) > 0 else 0
        avg_temp = temp_clean.mean() if len(temp_clean) > 0 else 0
        
        # Calculate previous period data (7 days ago)
        cutoff_date = datetime.now() - timedelta(days=7)
        df_prev = df[df['datetime'] < cutoff_date]
        
        if len(df_prev) > 0:
            prev_current = df_prev['current'].dropna().mean()
            prev_temp = df_prev['temperature'].dropna().mean()
            
            current_trend = ((avg_current - prev_current) / prev_current * 100) if prev_current > 0 else 0
            temp_trend = ((avg_temp - prev_temp) / prev_temp * 100) if prev_temp > 0 else 0
        else:
            current_trend = 0
            temp_trend = 0
        
        # Fault calculations
        total_faults = len(df[df['fault'] != 'Normal'])
        fault_rate = (total_faults / len(df) * 100) if len(df) > 0 else 0
        
        # Health index
        health_index = calculate_health_score_simple(df)
        health_status = get_health_status(health_index)
        
        return {
            'avg_current': round(float(avg_current), 2),
            'avg_temp': round(float(avg_temp), 2),
            'total_faults': int(total_faults),
            'fault_rate': round(float(fault_rate), 2),
            'current_trend': round(float(current_trend), 2),
            'temp_trend': round(float(temp_trend), 2),
            'health_index': round(float(health_index), 2),
            'health_status': str(health_status)
        }
    except Exception as e:
        print(f"Error in calculate_advanced_kpis: {e}")
        return {
            'avg_current': 0,
            'avg_temp': 0,
            'total_faults': 0,
            'fault_rate': 0,
            'current_trend': 0,
            'temp_trend': 0,
            'health_index': 0,
            'health_status': 'Unknown'
        }

def calculate_trend_analysis(df):
    """Calculate trend analysis with error handling"""
    try:
        # Group by date
        df['date'] = df['datetime'].dt.date
        daily_avg = df.groupby('date').agg({
            'temperature': 'mean',
            'current': 'mean'
        }).reset_index()
        
        # Convert dates to strings
        dates = [str(d) for d in daily_avg['date'].tolist()]
        temperatures = [float(x) if pd.notna(x) else 0 for x in daily_avg['temperature'].tolist()]
        currents = [float(x) if pd.notna(x) else 0 for x in daily_avg['current'].tolist()]
        
        return {
            'dates': dates,
            'temperatures': temperatures,
            'currents': currents
        }
    except Exception as e:
        print(f"Error in calculate_trend_analysis: {e}")
        return {'dates': [], 'temperatures': [], 'currents': []}

def calculate_correlation_matrix(df):
    """Calculate correlation matrix with error handling"""
    try:
        # Select numeric columns
        numeric_df = df[['temperature', 'current', 'vibration', 'voltage']].dropna()
        
        if len(numeric_df) < 2:
            return {'variables': ['temperature', 'current', 'vibration', 'voltage'], 'matrix': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
        
        corr_matrix = numeric_df.corr().values
        
        # Convert to list and handle NaN
        matrix = []
        for row in corr_matrix:
            matrix_row = [float(x) if not pd.isna(x) else 0 for x in row]
            matrix.append(matrix_row)
        
        return {
            'variables': ['temperature', 'current', 'vibration', 'voltage'],
            'matrix': matrix
        }
    except Exception as e:
        print(f"Error in calculate_correlation_matrix: {e}")
        return {'variables': ['temperature', 'current', 'vibration', 'voltage'], 
                'matrix': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}

def detect_anomalies(df):
    """Detect anomalies using statistical methods"""
    try:
        # Prepare data
        temp_data = df['temperature'].dropna()
        
        if len(temp_data) < 10:
            return {
                'dates': df['datetime'].dt.strftime('%Y-%m-%d').tolist()[:50],
                'values': df['temperature'].fillna(0).tolist()[:50],
                'anomaly_dates': [],
                'anomaly_values': [],
                'anomaly_descriptions': [],
                'anomaly_count': 0
            }
        
        # Calculate Z-scores
        mean_val = temp_data.mean()
        std_val = temp_data.std()
        
        if std_val == 0:
            z_scores = np.zeros(len(temp_data))
        else:
            z_scores = np.abs((temp_data - mean_val) / std_val)
        
        threshold = 2.5
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        # Get anomaly data
        anomaly_dates = []
        anomaly_values = []
        anomaly_descriptions = []
        
        for idx in anomaly_indices[:20]:  # Limit to 20 anomalies
            if idx < len(df):
                anomaly_dates.append(df.iloc[idx]['datetime'].strftime('%Y-%m-%d %H:%M:%S'))
                val = temp_data.iloc[idx]
                anomaly_values.append(float(val))
                anomaly_descriptions.append(f"Temperature anomaly: {val:.1f}°C")
        
        # Prepare full series
        dates = df['datetime'].dt.strftime('%Y-%m-%d').tolist()[:100]
        values = df['temperature'].fillna(0).tolist()[:100]
        
        return {
            'dates': dates,
            'values': values,
            'anomaly_dates': anomaly_dates,
            'anomaly_values': anomaly_values,
            'anomaly_descriptions': anomaly_descriptions,
            'anomaly_count': len(anomaly_indices)
        }
    except Exception as e:
        print(f"Error in detect_anomalies: {e}")
        return {
            'dates': [],
            'values': [],
            'anomaly_dates': [],
            'anomaly_values': [],
            'anomaly_descriptions': [],
            'anomaly_count': 0
        }

def generate_forecast(df):
    """Generate simple forecast using moving average"""
    try:
        # Prepare daily data
        df['date'] = df['datetime'].dt.date
        daily_temp = df.groupby('date')['temperature'].mean().reset_index()
        daily_temp = daily_temp.dropna()
        
        if len(daily_temp) < 7:
            return {
                'historical_dates': [],
                'historical_values': [],
                'forecast_dates': [],
                'forecast_values': [],
                'upper_bound': [],
                'lower_bound': []
            }
        
        # Use simple moving average for forecast
        window = min(7, len(daily_temp))
        last_values = daily_temp['temperature'].tail(window).tolist()
        
        # Simple forecast: average of last 'window' days
        forecast_avg = sum(last_values) / len(last_values)
        forecast_std = np.std(last_values) if len(last_values) > 1 else 5
        
        # Generate forecast for next 7 days
        last_date = daily_temp['date'].iloc[-1]
        forecast_dates = []
        forecast_values = []
        upper_bound = []
        lower_bound = []
        
        for i in range(1, 8):
            forecast_dates.append((last_date + timedelta(days=i)).strftime('%Y-%m-%d'))
            forecast_values.append(forecast_avg)
            upper_bound.append(forecast_avg + forecast_std)
            lower_bound.append(max(0, forecast_avg - forecast_std))
        
        return {
            'historical_dates': daily_temp['date'].astype(str).tolist(),
            'historical_values': daily_temp['temperature'].tolist(),
            'forecast_dates': forecast_dates,
            'forecast_values': forecast_values,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound
        }
    except Exception as e:
        print(f"Error in generate_forecast: {e}")
        return {
            'historical_dates': [],
            'historical_values': [],
            'forecast_dates': [],
            'forecast_values': [],
            'upper_bound': [],
            'lower_bound': []
        }

def get_fault_distribution(df):
    """Get fault distribution"""
    try:
        fault_df = df[df['fault'] != 'Normal']
        fault_counts = fault_df['fault'].value_counts()
        
        # Limit to top 6 fault types
        top_faults = fault_counts.head(6)
        
        return {
            'types': top_faults.index.tolist(),
            'counts': top_faults.values.tolist()
        }
    except Exception as e:
        print(f"Error in get_fault_distribution: {e}")
        return {'types': [], 'counts': []}

def calculate_health_score_simple(df):
    """Calculate simple health score"""
    try:
        # Normalize values
        temp_avg = df['temperature'].mean()
        current_avg = df['current'].mean()
        vib_avg = df['vibration'].mean()
        
        # Ideal ranges
        temp_score = max(0, 100 - (temp_avg / 2)) if temp_avg > 0 else 50
        current_score = max(0, 100 - (current_avg / 20)) if current_avg > 0 else 50
        vib_score = max(0, 100 - (vib_avg * 10)) if vib_avg > 0 else 50
        
        # Fault penalty
        fault_rate = len(df[df['fault'] != 'Normal']) / len(df) if len(df) > 0 else 0
        fault_penalty = fault_rate * 100
        
        # Weighted score
        health_score = (temp_score * 0.4 + current_score * 0.3 + vib_score * 0.3) * (1 - fault_penalty/100)
        
        return max(0, min(100, health_score))
    except Exception as e:
        print(f"Error in calculate_health_score_simple: {e}")
        return 50

def calculate_health_index_advanced(df):
    """Calculate advanced health index"""
    try:
        current_score = calculate_health_score_simple(df)
        
        # Calculate previous score (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        df_prev = df[df['datetime'] < cutoff_date]
        
        if len(df_prev) > 0:
            prev_score = calculate_health_score_simple(df_prev)
        else:
            prev_score = current_score
        
        # Calculate trend
        trend = current_score - prev_score
        
        # Determine warning threshold (70% of ideal)
        warning_threshold = 70
        
        return {
            'current_score': round(float(current_score), 1),
            'previous_score': round(float(prev_score), 1),
            'trend': round(float(trend), 1),
            'warning_threshold': warning_threshold
        }
    except Exception as e:
        print(f"Error in calculate_health_index_advanced: {e}")
        return {
            'current_score': 50,
            'previous_score': 50,
            'trend': 0,
            'warning_threshold': 70
        }

def get_health_status(score):
    """Get health status based on score"""
    if score >= 80:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    elif score >= 40:
        return 'Fair'
    elif score >= 20:
        return 'Poor'
    else:
        return 'Critical'

def analyze_seasonal_patterns(df):
    """Analyze seasonal patterns"""
    try:
        # Add month if not present
        if 'month' not in df.columns:
            df['month'] = df['datetime'].dt.month
        
        monthly_avg = df.groupby('month').agg({
            'temperature': 'mean',
            'current': 'mean'
        }).reset_index()
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Ensure all months are represented
        temperatures = []
        currents = []
        month_names = []
        
        for m in range(1, 13):
            month_data = monthly_avg[monthly_avg['month'] == m]
            if len(month_data) > 0:
                month_names.append(months[m-1])
                temperatures.append(float(month_data['temperature'].iloc[0]) if pd.notna(month_data['temperature'].iloc[0]) else 0)
                currents.append(float(month_data['current'].iloc[0]) if pd.notna(month_data['current'].iloc[0]) else 0)
            else:
                month_names.append(months[m-1])
                temperatures.append(0)
                currents.append(0)
        
        return {
            'months': month_names,
            'temperatures': temperatures,
            'currents': currents
        }
    except Exception as e:
        print(f"Error in analyze_seasonal_patterns: {e}")
        return {'months': [], 'temperatures': [], 'currents': []}

def analyze_hourly_patterns(df):
    """Analyze hourly patterns"""
    try:
        # Add hour if not present
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        
        hourly_avg = df.groupby('hour')['current'].mean().reset_index()
        
        # Ensure all hours are represented
        currents = []
        hours = list(range(24))
        
        for h in hours:
            hour_data = hourly_avg[hourly_avg['hour'] == h]
            if len(hour_data) > 0:
                currents.append(float(hour_data['current'].iloc[0]) if pd.notna(hour_data['current'].iloc[0]) else 0)
            else:
                currents.append(0)
        
        return {
            'hours': hours,
            'currents': currents
        }
    except Exception as e:
        print(f"Error in analyze_hourly_patterns: {e}")
        return {'hours': list(range(24)), 'currents': [0]*24}

def assess_risks(df):
    """Assess risks for each location"""
    try:
        # Check if location column exists
        if 'location' not in df.columns:
            locations = ['All']
            loc_df = df
        else:
            locations = df['location'].unique().tolist()
        
        risks = []
        
        for loc in locations:
            if loc == 'All':
                loc_df = df
                name = 'Overall'
            else:
                loc_df = df[df['location'] == loc]
                name = loc
            
            if len(loc_df) == 0:
                continue
            
            # Calculate risks
            temp_risk = min(100, (loc_df['temperature'].mean() / 100) * 100) if loc_df['temperature'].mean() > 0 else 0
            current_risk = min(100, (loc_df['current'].mean() / 1000) * 100) if loc_df['current'].mean() > 0 else 0
            vib_risk = min(100, (loc_df['vibration'].mean() / 10) * 100) if loc_df['vibration'].mean() > 0 else 0
            fault_risk = (len(loc_df[loc_df['fault'] != 'Normal']) / len(loc_df)) * 100 if len(loc_df) > 0 else 0
            overall_risk = (temp_risk + current_risk + vib_risk + fault_risk) / 4
            
            risks.append({
                'name': name,
                'values': [round(temp_risk, 1), round(current_risk, 1), round(vib_risk, 1), round(fault_risk, 1), round(overall_risk, 1)],
                'color': get_risk_color(overall_risk)
            })
        
        return {'risks': risks}
    except Exception as e:
        print(f"Error in assess_risks: {e}")
        return {'risks': []}

def get_risk_color(risk):
    """Get color based on risk level"""
    if risk >= 70:
        return '#ef4444'
    elif risk >= 40:
        return '#f59e0b'
    else:
        return '#10b981'

def generate_advanced_insights(df, kpis, anomalies, health):
    """Generate advanced insights and recommendations"""
    insights = []
    
    try:
        # Temperature insights
        temp_avg = df['temperature'].mean()
        temp_max = df['temperature'].max()
        
        if pd.notna(temp_max) and temp_max > 90:
            insights.append({
                'priority': 'High',
                'insight': f'Critical temperature spike detected (max: {temp_max:.1f}°C)',
                'impact': 'Potential transformer damage and reduced lifespan',
                'recommendation': 'Immediate inspection required. Check cooling systems and load distribution.',
                'confidence': 95
            })
        elif pd.notna(temp_avg) and temp_avg > 75:
            insights.append({
                'priority': 'Medium',
                'insight': f'High average temperature ({temp_avg:.1f}°C)',
                'impact': 'Accelerated insulation degradation',
                'recommendation': 'Schedule maintenance check. Consider load reduction during peak hours.',
                'confidence': 85
            })
        
        # Current insights
        current_avg = df['current'].mean()
        current_max = df['current'].max()
        
        if pd.notna(current_max) and current_max > 900:
            insights.append({
                'priority': 'High',
                'insight': f'Current overload detected (max: {current_max:.1f}A)',
                'impact': 'Risk of circuit breaker tripping and equipment damage',
                'recommendation': 'Redistribute load immediately. Investigate cause of overload.',
                'confidence': 90
            })
        elif pd.notna(current_avg) and current_avg > 700:
            insights.append({
                'priority': 'Medium',
                'insight': f'Sustained high current ({current_avg:.1f}A average)',
                'impact': 'Increased energy losses and thermal stress',
                'recommendation': 'Optimize load distribution and monitor closely.',
                'confidence': 80
            })
        
        # Fault insights
        if kpis['fault_rate'] > 5:
            insights.append({
                'priority': 'High',
                'insight': f'High fault rate detected: {kpis["fault_rate"]:.1f}%',
                'impact': 'Reduced reliability and potential system failures',
                'recommendation': 'Conduct root cause analysis. Implement predictive maintenance.',
                'confidence': 88
            })
        elif kpis['fault_rate'] > 2:
            insights.append({
                'priority': 'Medium',
                'insight': f'Elevated fault rate: {kpis["fault_rate"]:.1f}%',
                'impact': 'Increasing maintenance costs and downtime risk',
                'recommendation': 'Review maintenance schedule and fault patterns.',
                'confidence': 75
            })
        
        # Anomaly insights
        if anomalies['anomaly_count'] > 0:
            insights.append({
                'priority': 'Medium',
                'insight': f'{anomalies["anomaly_count"]} anomalies detected in the data',
                'impact': 'Unusual operating conditions requiring investigation',
                'recommendation': 'Analyze anomaly patterns. Check for external factors or equipment issues.',
                'confidence': 82
            })
        
        # Health insights
        if health['current_score'] < 40:
            insights.append({
                'priority': 'High',
                'insight': f'Critical health score: {health["current_score"]:.1f}%',
                'impact': 'High risk of failure and unplanned downtime',
                'recommendation': 'Immediate maintenance required. Consider replacement planning.',
                'confidence': 92
            })
        elif health['current_score'] < 60:
            insights.append({
                'priority': 'Medium',
                'insight': f'Degraded health score: {health["current_score"]:.1f}%',
                'impact': 'Reduced efficiency and increased failure probability',
                'recommendation': 'Schedule comprehensive maintenance. Monitor trends closely.',
                'confidence': 85
            })
        
        # Performance insights
        if kpis['current_trend'] > 10:
            insights.append({
                'priority': 'Medium',
                'insight': f'Significant upward current trend (+{kpis["current_trend"]:.1f}%)',
                'impact': 'Potential system overload if trend continues',
                'recommendation': 'Investigate load growth. Plan capacity upgrades if trend persists.',
                'confidence': 78
            })
        
        if kpis['temp_trend'] > 5:
            insights.append({
                'priority': 'Medium',
                'insight': f'Temperature trend increasing (+{kpis["temp_trend"]:.1f}%)',
                'impact': 'Accelerated aging of insulation system',
                'recommendation': 'Check cooling efficiency. Verify ambient temperature conditions.',
                'confidence': 80
            })
        
    except Exception as e:
        print(f"Error generating insights: {e}")
        # Add fallback insight
        insights.append({
            'priority': 'Low',
            'insight': 'System operating within normal parameters',
            'impact': 'No immediate concerns detected',
            'recommendation': 'Continue regular monitoring and maintenance schedule.',
            'confidence': 95
        })
    
    # Ensure we always have at least one insight
    if len(insights) == 0:
        insights.append({
            'priority': 'Low',
            'insight': 'System operating within normal parameters',
            'impact': 'No immediate concerns detected',
            'recommendation': 'Continue regular monitoring and maintenance schedule.',
            'confidence': 95
        })
    
    # Sort by priority
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    insights.sort(key=lambda x: priority_order.get(x['priority'], 3))
    
    return insights[:10]

@app.route('/api/analytics/data')
def analytics_data():
    """API endpoint for analytics data"""
    location = request.args.get('location', 'Kigali')
    period = request.args.get('period', '30')  # days
    
    try:
        days = int(period)
        start_date = datetime.now() - timedelta(days=days)
        
        # Get daily summaries
        summaries = DailySummary.query.filter(
            DailySummary.location == location,
            DailySummary.date >= start_date.date()
        ).order_by(DailySummary.date).all()
        
        # Prepare data for charts
        dates = [s.date.strftime('%Y-%m-%d') for s in summaries]
        temperatures = [s.avg_temperature for s in summaries]
        currents = [s.avg_current for s in summaries]
        faults = [s.fault_count for s in summaries]
        
        # Get fault distribution
        fault_dist = db.session.query(
            TransformerReading.fault,
            db.func.count(TransformerReading.id).label('count')
        ).filter(
            TransformerReading.location == location,
            TransformerReading.fault != 'Normal'
        ).group_by(TransformerReading.fault).all()
        
        # Get hourly patterns
        hourly_data = db.session.query(
            TransformerReading.hour,
            db.func.avg(TransformerReading.current).label('avg_current'),
            db.func.avg(TransformerReading.temperature).label('avg_temp')
        ).filter(
            TransformerReading.location == location,
            TransformerReading.reading_datetime >= start_date
        ).group_by(TransformerReading.hour).order_by(TransformerReading.hour).all()
        
        return jsonify({
            'success': True,
            'dates': dates,
            'temperatures': temperatures,
            'currents': currents,
            'faults': faults,
            'fault_distribution': [{'name': f[0], 'count': f[1]} for f in fault_dist],
            'hourly_pattern': [{'hour': h[0], 'current': h[1], 'temp': h[2]} for h in hourly_data]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/faults')
def faults():
    """Fault analysis page"""
    page = request.args.get('page', 1, type=int)
    location = request.args.get('location', '')
    
    # Build query
    query = TransformerReading.query.filter(TransformerReading.fault != 'Normal')
    
    if location and location != 'all':
        query = query.filter_by(location=location)
    
    # Paginate
    pagination = query.order_by(
        TransformerReading.reading_datetime.desc()
    ).paginate(page=page, per_page=50, error_out=False)
    
    # Get summary statistics
    total_faults = query.count()
    fault_by_type = db.session.query(
        TransformerReading.fault,
        db.func.count(TransformerReading.id).label('count')
    ).filter(TransformerReading.fault != 'Normal')
    
    if location and location != 'all':
        fault_by_type = fault_by_type.filter_by(location=location)
    
    fault_by_type = fault_by_type.group_by(TransformerReading.fault).all()
    
    return render_template(
        'faults.html',
        faults=pagination.items,
        pagination=pagination,
        locations=LOCATIONS,
        current_location=location,
        total_faults=total_faults,
        fault_by_type=fault_by_type
    )

@app.route('/api/readings', methods=['GET'])
def api_readings():
    """REST API for readings"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    query = TransformerReading.query.order_by(TransformerReading.reading_datetime.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'data': [r.to_dict() for r in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    })

@app.route('/api/readings/<int:id>', methods=['GET'])
def api_reading(id):
    """REST API for single reading"""
    reading = TransformerReading.query.get_or_404(id)
    return jsonify(reading.to_dict())

@app.route('/api/summary/daily')
def api_daily_summary():
    """API for daily summary"""
    location = request.args.get('location')
    limit = request.args.get('limit', 30, type=int)
    
    query = DailySummary.query.order_by(DailySummary.date.desc())
    
    if location:
        query = query.filter_by(location=location)
    
    summaries = query.limit(limit).all()
    return jsonify([s.to_dict() for s in summaries])

@app.route('/api/stats')
def api_stats():
    """API for overall statistics"""
    total_readings = TransformerReading.query.count()
    total_faults = TransformerReading.query.filter(TransformerReading.fault != 'Normal').count()
    
    return jsonify({
        'total_readings': total_readings,
        'total_faults': total_faults,
        'fault_rate': round((total_faults / total_readings * 100), 2) if total_readings > 0 else 0,
        'locations': len(LOCATIONS)
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Create tables and initialize database
def init_db():
    """Initialize database with tables"""
    with app.app_context():
        db.create_all()
        print("Database initialized")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)