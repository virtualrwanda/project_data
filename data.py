import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import gc
import os
import shutil
from collections import Counter

np.random.seed(42)

# ============================
# SETTINGS (MEMORY SAFE)
# ============================
CHUNK_SIZE = 50000
STEP_SECONDS = 60

# ============================
# DATASET SIZE OPTIONS
# ============================
print("\n📊 Dataset Size Options:")
print("1. Small (1 month) - ~50MB")
print("2. Medium (6 months) - ~300MB")
print("3. Large (1 year) - ~600MB")
print("4. Full (7 years) - ~4GB")
print("5. Custom")

size_choice = input("\nChoose dataset size (1-5): ").strip()

if size_choice == '1':
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    print("📦 Generating SMALL dataset (1 month)")
elif size_choice == '2':
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    print("📦 Generating MEDIUM dataset (6 months)")
elif size_choice == '3':
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    print("📦 Generating LARGE dataset (1 year)")
elif size_choice == '4':
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 3, 31)
    print("📦 Generating FULL dataset (7 years)")
else:
    print("Enter custom dates:")
    year = int(input("Start year (2024): ") or "2024")
    month = int(input("Start month (1-12): "))
    day = int(input("Start day: "))
    start_date = datetime(year, month, day)
    
    year = int(input("End year: "))
    month = int(input("End month: "))
    day = int(input("End day: "))
    end_date = datetime(year, month, day)
    print(f"📦 Generating CUSTOM dataset from {start_date.date()} to {end_date.date()}")

OUTPUT_FILE = "transformer_dataset.csv.gz"
METADATA_FILE = "metadata.json"

# ============================
# DISK SPACE CHECK
# ============================
def check_disk_space(min_gb=2):
    """Check if there's enough disk space"""
    total, used, free = shutil.disk_usage(".")
    free_gb = free // (2**30)
    print(f"💾 Free disk space: {free_gb} GB")
    
    if free_gb < min_gb:
        print(f"⚠️  WARNING: Only {free_gb} GB free. Minimum recommended: {min_gb} GB")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("❌ Dataset generation cancelled.")
            exit(1)
    return free_gb

# ============================
# TRANSFORMERS CONFIGURATION
# ============================
locations = ['Kigali', 'Bugesera', 'Rulindo', 'Gicumbi']

transformers = {
    'Kigali':   {'serial': 'TRF-2024-KGL-001', 'base_current': 60.0},
    'Bugesera': {'serial': 'TRF-2024-BGS-001', 'base_current': 40.0},
    'Rulindo':  {'serial': 'TRF-2024-RLD-001', 'base_current': 30.0},
    'Gicumbi':  {'serial': 'TRF-2024-GCM-001', 'base_current': 50.0}
}

# ============================
# DATE GENERATOR (FIXED WARNING)
# ============================
def date_range_generator(start, end, step_seconds, chunk_size):
    """Generate date ranges in chunks to avoid memory issues"""
    current = start
    while current <= end:
        # FIXED: Use lowercase 's' instead of 'S' for seconds
        chunk = pd.date_range(start=current, periods=chunk_size, freq=f"{step_seconds}s")
        current = chunk[-1] + timedelta(seconds=step_seconds)
        yield chunk

# ============================
# DATA GENERATION
# ============================
def generate_data(location, specs, date_chunk, first_chunk):
    """Generate synthetic transformer data for a chunk of timestamps"""
    try:
        size = len(date_chunk)
        base_current = specs['base_current']

        # Time features for patterns
        hour = date_chunk.hour.values + date_chunk.minute.values / 60
        day = date_chunk.dayofyear.values

        # Daily and seasonal patterns
        cycle = 0.2 * np.sin(2 * np.pi * hour / 24) + 0.1 * np.sin(2 * np.pi * day / 365)

        # ============================
        # CURRENT
        # ============================
        current = base_current * (1 + cycle + np.random.normal(0, 0.05, size))
        current = np.clip(current, 0, 120).astype(np.float32)

        # ============================
        # LOAD STATUS
        # ============================
        load_status = np.empty(size, dtype="U10")
        load_status[current <= 21.7] = 'under'
        load_status[(current > 21.7) & (current <= 57.8)] = 'normal'
        load_status[(current > 57.8) & (current <= 72.2)] = 'heavy'
        load_status[current > 72.2] = 'over'

        # ============================
        # TEMPERATURE
        # ============================
        temperature = np.where(load_status == 'under',
                              np.random.uniform(40, 55, size),
                       np.where(load_status == 'normal',
                              np.random.uniform(55, 65, size),
                       np.where(load_status == 'heavy',
                              np.random.uniform(65, 85, size),
                              np.random.uniform(85, 120, size))))
        temperature = (temperature + 3 * cycle).astype(np.float32)

        # ============================
        # VIBRATION
        # ============================
        vibration = np.where(load_status == 'under',
                            np.random.uniform(0, 2, size),
                     np.where(load_status == 'normal',
                            np.random.uniform(2, 4, size),
                     np.where(load_status == 'heavy',
                            np.random.uniform(4, 6, size),
                            np.random.uniform(6, 10, size))))
        vibration = (vibration + 0.3 * cycle).astype(np.float32)

        # ============================
        # VOLTAGE
        # ============================
        voltage = np.where(load_status == 'over',
                           np.random.uniform(200, 260, size),
                           np.random.uniform(220, 240, size)).astype(np.float32)

        # ============================
        # POWER
        # ============================
        power = (voltage * current / 1000).astype(np.float32)

        # ============================
        # WARNINGS
        # ============================
        warning = np.empty(size, dtype="U20")
        warning[:] = 'None'
        warning[(temperature > 60) & (temperature <= 65)] = 'Temp Rising'

        # ============================
        # FAULTS
        # ============================
        fault = np.empty(size, dtype="U30")
        fault[:] = 'Normal'
        
        # Fault conditions (ordered by priority)
        fault[(temperature > 95)] = 'Critical Overheat'
        fault[(temperature > 85)] = 'Overheat'
        fault[(temperature > 65) & (load_status == 'normal')] = 'Early Overheat'
        fault[(vibration > 6)] = 'Mechanical looseness'
        fault[(voltage < 220) | (voltage > 240)] = 'Voltage abnormal'
        fault[(load_status == 'over') & (temperature > 85)] = 'Overload + Thermal'
        fault[(load_status == 'over')] = 'Overload'

        # ============================
        # DATAFRAME
        # ============================
        df = pd.DataFrame({
            'serial_number': specs['serial'],
            'location': location,
            'reading_datetime': date_chunk,
            'current': current,
            'voltage': voltage,
            'temperature': temperature,
            'vibration': vibration,
            'power_kw': power,
            'load_status': load_status,
            'warning': warning,
            'fault': fault
        })

        # ============================
        # WRITE (DIRECT GZIP)
        # ============================
        df.to_csv(
            OUTPUT_FILE,
            mode='a' if not first_chunk else 'w',
            header=first_chunk,
            index=False,
            compression='gzip'
        )

        # ============================
        # FREE MEMORY
        # ============================
        del df, current, voltage, temperature, vibration, power, load_status, warning, fault
        gc.collect()
        
        return True
        
    except OSError as e:
        if e.errno == 28:  # No space left on device
            print(f"\n❌ ERROR: No space left on device!")
            print(f"   The dataset is too large for your available disk space.")
            print(f"   Try generating a smaller dataset.")
            return False
        else:
            raise e

# ============================
# METADATA GENERATION (CHUNKED)
# ============================
def generate_metadata():
    """Generate metadata statistics without loading entire dataset"""
    print("\n📊 Generating metadata...")
    
    fault_counter = Counter()
    load_counter = Counter()
    warning_counter = Counter()
    total = 0
    
    try:
        # Read in chunks to avoid memory issues
        for chunk in pd.read_csv(OUTPUT_FILE, chunksize=100000, compression='gzip'):
            total += len(chunk)
            fault_counter.update(chunk['fault'].fillna('None'))
            load_counter.update(chunk['load_status'].fillna('Unknown'))
            warning_counter.update(chunk['warning'].fillna('None'))
            
            # Progress indicator
            if total % 1000000 == 0:
                print(f"   Processed {total:,} records...")
    except FileNotFoundError:
        print("   No data file found to generate metadata.")
        return None
    
    # Calculate dataset size
    file_size_bytes = os.path.getsize(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
    
    metadata = {
        "dataset_name": "Transformer Monitoring Dataset",
        "date_range": f"{start_date.date()} to {end_date.date()}",
        "interval_seconds": STEP_SECONDS,
        "normal_temperature_range": "<65°C",
        "total_records": total,
        "file_size": {
            "bytes": file_size_bytes,
            "mb": round(file_size_mb, 2),
            "gb": round(file_size_gb, 2)
        },
        "fault_distribution": dict(fault_counter),
        "load_distribution": dict(load_counter),
        "warning_distribution": dict(warning_counter),
        "transformers": {
            loc: {
                "serial": specs['serial'],
                "base_current": specs['base_current']
            }
            for loc, specs in transformers.items()
        },
        "generated_at": datetime.now().isoformat()
    }
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

# ============================
# MAIN EXECUTION
# ============================
def main():
    """Main execution function"""
    print("🚀 Transformer Dataset Generator")
    print("=" * 50)
    
    # Check disk space (lower requirement for smaller datasets)
    free_gb = check_disk_space(min_gb=1)
    
    # Remove existing output file if it exists
    if os.path.exists(OUTPUT_FILE):
        print(f"🗑️  Removing existing file: {OUTPUT_FILE}")
        os.remove(OUTPUT_FILE)
    
    # Calculate estimated size
    total_minutes = int((end_date - start_date).total_seconds() / 60)
    total_records = total_minutes * len(locations)
    estimated_size_mb = (total_records * 200) / (1024 * 1024)  # Rough estimate: 200 bytes per record
    print(f"📈 Estimated records: {total_records:,}")
    print(f"💿 Estimated file size: ~{estimated_size_mb:.1f} MB (compressed)")
    
    # Check if estimated size exceeds available space
    if estimated_size_mb > (free_gb * 1024):
        print(f"\n⚠️  WARNING: Estimated dataset size ({estimated_size_mb:.1f} MB) exceeds available disk space ({free_gb} GB)!")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("❌ Generation cancelled.")
            return
    
    print()
    response = input("Start generating dataset? (y/n): ")
    if response.lower() != 'y':
        print("❌ Generation cancelled.")
        return
    
    print("\n🔄 Generating dataset...")
    first_chunk = True
    total_chunks = 0
    
    for location in locations:
        specs = transformers[location]
        date_gen = date_range_generator(start_date, end_date, STEP_SECONDS, CHUNK_SIZE)
        
        chunk_count = 0
        for i, date_chunk in enumerate(date_gen):
            success = generate_data(location, specs, date_chunk, first_chunk)
            if not success:
                print(f"\n❌ Generation stopped due to disk space error.")
                print(f"   Partial dataset saved up to {location} - chunk {i}")
                return
            
            first_chunk = False
            chunk_count += 1
            total_chunks += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   {location}: chunk {i+1} | Total chunks: {total_chunks}")
        
        print(f"✅ {location} completed ({chunk_count} chunks)")
    
    print("\n📦 Dataset generation complete!")
    
    # Generate metadata
    metadata = generate_metadata()
    
    if metadata:
        # Print summary
        print("\n" + "=" * 50)
        print("✅ DATASET COMPLETE")
        print("=" * 50)
        print(f"📁 File: {OUTPUT_FILE}")
        print(f"📊 Total Records: {metadata['total_records']:,}")
        print(f"💾 File Size: {metadata['file_size']['mb']:.2f} MB ({metadata['file_size']['gb']:.2f} GB)")
        print(f"📋 Metadata: {METADATA_FILE}")
        print("\n📈 Quick Statistics:")
        if metadata['fault_distribution']:
            most_common_fault = max(metadata['fault_distribution'].items(), key=lambda x: x[1])
            print(f"   Most common fault: {most_common_fault[0]} ({most_common_fault[1]:,})")
        if metadata['load_distribution']:
            most_common_load = max(metadata['load_distribution'].items(), key=lambda x: x[1])
            print(f"   Most common load: {most_common_load[0]} ({most_common_load[1]:,})")
        if metadata['warning_distribution']:
            most_common_warning = max(metadata['warning_distribution'].items(), key=lambda x: x[1])
            print(f"   Most common warning: {most_common_warning[0]} ({most_common_warning[1]:,})")
    
    print("\n✅ Generation finished successfully!")

# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user.")
        print("Partial dataset may have been saved.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()