import pandas as pd
import numpy as np
from config import raw_bandwidth

def load_and_preprocess_data():
    """Load and preprocess the raw bandwidth data"""
    # Clean the raw data (replace -1 with NaN, then forward fill)
    cleaned_bandwidth = []
    for value in raw_bandwidth:
        if value == -1:
            cleaned_bandwidth.append(np.nan)
        else:
            cleaned_bandwidth.append(value)

    # Create DataFrame with the raw data
    df = pd.DataFrame({'bandwidth_mbps': cleaned_bandwidth})
    df['bandwidth_mbps'] = df['bandwidth_mbps'].ffill()

    # Create timestamps (assuming regular intervals)
    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5min')
    df.set_index('timestamp', inplace=True)

    print(f"📊 Using Raw Bandwidth Data:")
    print(f"   Total data points: {len(df)}")
    print(f"   Bandwidth range: {df['bandwidth_mbps'].min():.2f} - {df['bandwidth_mbps'].max():.2f} Mbps")
    print(f"   Average bandwidth: {df['bandwidth_mbps'].mean():.2f} Mbps")
    
    return df
