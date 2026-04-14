import xarray as xr
from scipy import stats
import os
import numpy as np
import pandas as pd

# 1. DYNAMIC PATH CONFIGURATION 
current_dir = os.path.dirname(os.path.abspath(__file__))
nc_path = os.path.normpath(os.path.join(current_dir, "..", "Data", "global_warming_data.nc"))

try:
    # 2. DATA LOADING 
    ds = xr.open_dataset(nc_path)
    istanbul = ds['t2m'].sel(latitude=41.0, longitude=29.0, method='nearest') - 273.15

    # 3. VOLATILITY TREND ANALYSIS (Spearman Rank Correlation) 
    # Standard deviation per year represents "Atmospheric Chaos"
    yearly_vol = istanbul.groupby('valid_time.year').std()
    years = yearly_vol.year.values
    vol_values = yearly_vol.values
    
    # Spearman Rho measures if the chaos is increasing over time
    correlation, p_trend = stats.spearmanr(years, vol_values)

    # 4. RESULT OUTPUTS 
    print("\n" + "="*65)
    print("METEOROLOGICAL ANALYSIS: TECHNOLOGY VS. ATMOSPHERIC CHAOS")
    print("="*65)

    print(f"\n[STATISTICAL METRICS]")
    print(f" > Chaos Trend Intensity (Spearman Rho): {correlation:.3f}")
    print(f" > Statistical Significance (p-value):   {p_trend:.5f}")

    print("\n" + "="*65)
    print("TECHNOLOGICAL FEASIBILITY INTERPRETATION")
    print("="*65)

    if p_trend < 0.05:
        if correlation > 0:
            print("STATUS: ATMOSPHERE IS WINNING")
            print("RESULT: A significant upward trend in volatility detected.")
            print("IMPLICATION: As the atmosphere becomes more chaotic, current forecasting")
            print("models (like ECMWF) face a 'Predictability Crisis'. Technology must")
            print("evolve faster to maintain current accuracy levels.")
        else:
            print("STATUS: TECHNOLOGY IS WINNING")
            print("RESULT: Volatility is significantly decreasing.")
            print("IMPLICATION: The atmosphere is becoming more stable, making it easier")
            print("for deterministic models to provide highly accurate long-term results.")
    else:
        print("STATUS: BALANCED COMPETITION")
        print("RESULT: No significant trend in atmospheric volatility (p > 0.05).")
        print("IMPLICATION: Atmospheric chaos is stable. Any errors in forecasting are")
        print("likely due to local micro-anomalies rather than a systemic failure of")
        print("global meteorological technology. Current models remain highly reliable.")

    print("="*65 + "\n")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")