import cdsapi
import os

# Initialize the Copernicus Climate Data Store API client
client = cdsapi.Client()

# Define the period and seasonal months (January and July)
years = [str(y) for y in range(1990, 2025)]
months = [str(m).zfill(2) for m in range(1, 13)]

# Define the request for ERA5 monthly mean temperature at 2m
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["2m_temperature"],
    "year": years,
    "month": months,
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "..", "global_warming_data.nc")

print("Starting data collection...")
try:
    client.retrieve("reanalysis-era5-single-levels-monthly-means", request).download(save_path)
    print(f"SUCCESS: Data saved to {save_path}")
except Exception as e:
    print(f"Error: {e}")