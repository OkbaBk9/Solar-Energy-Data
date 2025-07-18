import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class NASAPowerSolarFetcher:
    def __init__(self):
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.locations = {
            'Tamanrasset': (5.5281, 22.7851),
            'Adrar': (0.2942, 27.8702),
            'Illizi': (8.4723, 26.4840),
            'In Salah': (2.4667, 27.2500),
            'Djanet': (9.4842, 24.5542)
        }
        
        # Comprehensive parameter sets for solar energy data science
        self.parameter_groups = {
            'core_solar': [
                'ALLSKY_SFC_SW_DWN',     # All Sky Global Horizontal Irradiance (GHI)
                'CLRSKY_SFC_SW_DWN',     # Clear Sky Global Horizontal Irradiance
                'ALLSKY_SFC_SW_DNI',     # All Sky Direct Normal Irradiance (DNI)
                'CLRSKY_SFC_SW_DNI',     # Clear Sky Direct Normal Irradiance
                'ALLSKY_SFC_SW_DIFF',    # All Sky Diffuse Horizontal Irradiance (DHI)
                'CLRSKY_SFC_SW_DIFF',    # Clear Sky Diffuse Horizontal Irradiance
            ],
            'atmospheric': [
                'CLOUD_AMT',             # Cloud Amount (%)
                'ALLSKY_SFC_UV_INDEX',   # UV Index
                'TOA_SW_DWN',           # Top of Atmosphere Shortwave Downward
                'ALLSKY_SFC_PAR_TOT',   # Photosynthetically Active Radiation
                'ALLSKY_SRF_ALB',       # All Sky Surface Albedo
            ],
            'temperature': [
                'T2M',                   # Temperature at 2 meters 
                'T2M_MAX',              # Maximum Temperature at 2 meters 
                'T2M_MIN',              # Minimum Temperature at 2 meters 
                'T2M_RANGE',            # Temperature Range at 2 meters 
                'T2MDEW',               # Dew Point Temperature at 2 meters
                'T2MWET',               # Wet Bulb Temperature at 2 meters 
                # All in (Celsius)
            ],
            'wind': [
                'WS2M',                 # Wind Speed at 2 meters (m/s)
                'WS2M_MAX',             # Maximum Wind Speed at 2 meters (m/s)
                'WS2M_MIN',             # Minimum Wind Speed at 2 meters (m/s)
                'WD2M',                 # Wind Direction at 2 meters (degrees)
                'WS10M',                # Wind Speed at 10 meters (m/s)
                'WS10M_MAX',            # Maximum Wind Speed at 10 meters (m/s)
            ],
            'pressure_humidity': [
                'PS',                   # Surface Pressure (kPa)
                'RH2M',                 # Relative Humidity at 2 meters (%)
                'QV2M',                 # Specific Humidity at 2 meters (g/kg)
            ],
            'precipitation': [
                'PRECTOTCORR',          # Precipitation Corrected (mm/day)
                'PRECTOTCORR_SUM',      # Precipitation Sum (mm)
            ],
            'solar_angles': [
                'SZA',                       # Solar Zenith Angle (degrees)
                'ALLSKY_SFC_SW_DWN_00_GMT',  # Solar radiation at specific times
                'ALLSKY_SFC_SW_DWN_03_GMT',
                'ALLSKY_SFC_SW_DWN_06_GMT',
                'ALLSKY_SFC_SW_DWN_09_GMT',
                'ALLSKY_SFC_SW_DWN_12_GMT',
                'ALLSKY_SFC_SW_DWN_15_GMT',
                'ALLSKY_SFC_SW_DWN_18_GMT',
                'ALLSKY_SFC_SW_DWN_21_GMT',
            ]
        }
    
    def fetch_daily_nasa_improved(self, location, year, params=None, community="re", max_params_per_request=10):
        if params is None:
            # Default comprehensive parameter set for solar analysis
            params = (self.parameter_groups['core_solar'] + 
                     self.parameter_groups['temperature'][:3] +
                     self.parameter_groups['atmospheric'][:3] +
                     self.parameter_groups['wind'][:3] +
                     self.parameter_groups['pressure_humidity'])
        
        lon, lat = location
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        # Split parameters into chunks to avoid API limits
        param_chunks = [params[i:i+max_params_per_request] for i in range(0, len(params), max_params_per_request)]
        all_dfs = []
        
        for chunk_idx, param_chunk in enumerate(param_chunks):
            print(f"Fetching parameter chunk {chunk_idx + 1}/{len(param_chunks)}: {param_chunk}")
            
            # Build the API URL
            url_params = {
                'parameters': ','.join(param_chunk),
                'community': community,
                'longitude': lon,
                'latitude': lat,
                'start': start_date,
                'end': end_date,
                'format': 'json'
            }
            
            try:
                response = requests.get(self.base_url, params=url_params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for errors in response
                if 'error' in data:
                    print(f"API Error: {data['error']}")
                    continue
                    
                # Extract parameter data
                if 'properties' not in data or 'parameter' not in data['properties']:
                    print("No parameter data found in response")
                    continue
                
                parameter_data = data['properties']['parameter']
                
                # Create DataFrame for this chunk
                df_data = {}
                dates = []
                
                # Get all available dates from the first parameter
                if parameter_data:
                    first_param = list(parameter_data.keys())[0]
                    for date_str, value in parameter_data[first_param].items():
                        try:
                            date = datetime.strptime(date_str, '%Y%m%d')
                            dates.append(date)
                        except ValueError:
                            continue
                
                # Sort dates
                dates.sort()
                
                # Extract data for each parameter
                for param in param_chunk:
                    if param in parameter_data:
                        values = []
                        for date in dates:
                            date_str = date.strftime('%Y%m%d')
                            value = parameter_data[param].get(date_str, np.nan)
                            # Convert -999 or other fill values to NaN
                            if value == -999 or value == -99:
                                value = np.nan
                            values.append(value)
                        df_data[param] = values
                    else:
                        print(f"Parameter {param} not found in response")
                        df_data[param] = [np.nan] * len(dates)
                
                # Create DataFrame for this chunk
                if dates:
                    chunk_df = pd.DataFrame(df_data, index=dates)
                    all_dfs.append(chunk_df)
                
                time.sleep(0.5)  # Small delay between chunks
                
            except requests.exceptions.RequestException as e:
                print(f"Request failed for chunk {chunk_idx + 1}: {e}")
                continue
            except json.JSONDecodeError as e:
                print(f"JSON decode error for chunk {chunk_idx + 1}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error for chunk {chunk_idx + 1}: {e}")
                continue
        
        # Combine all chunks
        if all_dfs:
            df = pd.concat(all_dfs, axis=1)
            return df
        
        return None
    
    def try_multiple_approaches(self, location_name, year, param_groups=None):
      
        location = self.locations[location_name]
        
        if param_groups is None:
            param_groups = ['core_solar', 'temperature', 'atmospheric']
        
        # Get parameters for specified groups
        test_params = []
        for group in param_groups:
            test_params.extend(self.parameter_groups.get(group, []))
        
        # Different approaches to try
        approaches = [
            {
                'community': 're',
                'params': test_params[:10],
                'description': 'RE community with comprehensive solar parameters'
            },
            {
                'community': 'ag',
                'params': ['ALLSKY_SFC_SW_DWN', 'T2M', 'RH2M', 'WS2M'],
                'description': 'AG community with basic parameters'
            },
            {
                'community': 'sb',
                'params': self.parameter_groups['core_solar'][:4] + ['T2M'],
                'description': 'SB community with core solar parameters'
            },
            {
                'community': 're',
                'params': self.parameter_groups['core_solar'][:3],
                'description': 'RE community with core solar parameters only'
            },
            {
                'community': 'ag',
                'params': ['T2M', 'RH2M'],
                'description': 'Basic connectivity test with temperature and humidity'
            }
        ]
        
        print(f"\nTesting different approaches for {location_name} ({year}) ")
        
        for i, approach in enumerate(approaches, 1):
            print(f"\nApproach {i}: {approach['description']}")
            print("-" * 50)
            
            df = self.fetch_daily_nasa_improved(
                location, 
                year, 
                params=approach['params'],
                community=approach['community']
            )
            
            if df is not None and not df.empty:
                # Check if we have valid solar data
                solar_params = [p for p in approach['params'] if 'SW' in p]
                if solar_params:
                    solar_param = solar_params[0]
                    if solar_param in df.columns:
                        valid_solar = df[solar_param].notna().sum()
                        if valid_solar > 0:
                            print(f"sucess! Found {valid_solar} valid solar irradiance values")
                            return df, approach
                
                # If no solar data but we have temperature, it's partial success
                if 'T2M' in df.columns and df['T2M'].notna().sum() > 0:
                    print(f"Partial success: Temperature data available")
                    if len(solar_params) == 0:
                        continue
            
            print("No valid data found")
            time.sleep(1)
        
        return None, None
    
    def create_sahara_solar_dataset(self, year=2020, dataset_type='comprehensive'):
        # Define parameter sets based on dataset type
        if dataset_type == 'basic':
            param_groups = ['core_solar', 'temperature']
        elif dataset_type == 'comprehensive':
            param_groups = ['core_solar', 'temperature', 'atmospheric', 'wind', 'pressure_humidity']
        elif dataset_type == 'full':
            param_groups = list(self.parameter_groups.keys())
        else:
            param_groups = ['core_solar', 'temperature', 'atmospheric']
        
        results = {}
        successful_approach = None
        
        print(f"Creating Algerian Sahara solar dataset for {year}")
        print(f"Dataset type: {dataset_type}")
        print("=" * 60)
        
        for location_name in self.locations:
            print(f"\nProcessing {location_name}...")
            
            if successful_approach is None:
                # Try different approaches for the first location
                df, approach = self.try_multiple_approaches(location_name, year, param_groups)
                if df is not None and approach is not None:
                    successful_approach = approach
                    results[location_name] = df
                    print(f"{location_name}: Using {approach['description']}")
                else:
                    print(f"{location_name}: No working approach found")
                    results[location_name] = None
            else:
                # Use the successful approach for remaining locations
                print(f"Using successful approach: {successful_approach['description']}")
                df = self.fetch_daily_nasa_improved(
                    self.locations[location_name], 
                    year,
                    params=successful_approach['params'],
                    community=successful_approach['community']
                )
                results[location_name] = df
                if df is not None:
                    print(f"{location_name}: Data retrieved successfully")
                else:
                    print(f"{location_name}: Data retrieval failed")
            
            time.sleep(1)  #  API rate limits
        
        return results, successful_approach
    
    def create_ml_ready_dataset(self, year=2020, include_derived_features=True):
        
        # Get comprehensive data
        all_data, approach = self.create_sahara_solar_dataset(year, 'comprehensive')
        
        if not any(df is not None for df in all_data.values()):
            print("No data retrieved for any location")
            return None
        
        # Combine all locations into a single dataset
        combined_data = []
        
        for location_name, df in all_data.items():
            if df is not None:
                # Add location information
                df_copy = df.copy()
                df_copy['location'] = location_name
                df_copy['longitude'] = self.locations[location_name][0]
                df_copy['latitude'] = self.locations[location_name][1]
                df_copy['date'] = df_copy.index
                df_copy = df_copy.reset_index(drop=True)
                
                if include_derived_features:
                    df_copy = self.add_derived_features(df_copy)
                
                combined_data.append(df_copy)
        
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            
            # Reorder columns for better readability
            id_cols = ['date', 'location', 'longitude', 'latitude']
            data_cols = [col for col in final_df.columns if col not in id_cols]
            final_df = final_df[id_cols + data_cols]
            
            print(f"\nML-ready dataset created:")
            print(f"Shape: {final_df.shape}")
            print(f"Locations: {final_df['location'].unique()}")
            print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
            print(f"Features: {len(data_cols)} data columns")
            
            return final_df
        
        return None
    
    def add_derived_features(self, df):
        """Add derived features useful for ML models"""
        df = df.copy()
        
        # Date-based features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                          3: 'Spring', 4: 'Spring', 5: 'Spring',
                                          6: 'Summer', 7: 'Summer', 8: 'Summer',
                                          9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
            df['is_summer'] = df['season'].eq('Summer').astype(int)
            df['is_winter'] = df['season'].eq('Winter').astype(int)
        
        # Solar-related derived features
        if 'ALLSKY_SFC_SW_DWN' in df.columns and 'CLRSKY_SFC_SW_DWN' in df.columns:
            df['clearness_index'] = df['ALLSKY_SFC_SW_DWN'] / df['CLRSKY_SFC_SW_DWN'].replace(0, np.nan)
            df['cloud_impact'] = df['CLRSKY_SFC_SW_DWN'] - df['ALLSKY_SFC_SW_DWN']
        
        # Temperature-related features
        if 'T2M_MAX' in df.columns and 'T2M_MIN' in df.columns:
            df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']
        
        if 'T2M' in df.columns:
            df['temp_category'] = pd.cut(df['T2M'], bins=[-np.inf, 20, 35, np.inf], 
                                       labels=['Cool', 'Moderate', 'Hot'])
        
        # Wind power density (approximate)
        if 'WS2M' in df.columns:
            df['wind_power_density'] = 0.5 * 1.225 * (df['WS2M'] ** 3)
        
        # Comfort indices
        if 'T2M' in df.columns and 'RH2M' in df.columns:
            # Heat Index approximation
            df['heat_index'] = df['T2M'] + 0.5 * (df['RH2M'] / 100) * (df['T2M'] - 14.4)
        
        # Moving averages for trend analysis
        solar_cols = [col for col in df.columns if 'SW_DWN' in col or 'SW_DNI' in col]
        for col in solar_cols:
            if col in df.columns:
                df[f'{col}_7day_avg'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_30day_avg'] = df[col].rolling(window=30, min_periods=1).mean()
        
        return df
    
    def save_to_csv(self, df, filename=None, location_name=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if location_name:
                filename = f"solar_data_{location_name}_{timestamp}.csv"
            else:
                filename = f"solar_data_sahara_{timestamp}.csv"
        
        # Format the data for CSV
        df_export = df.copy()
        
        # Convert date index to column if it's the index
        if isinstance(df_export.index, pd.DatetimeIndex):
            df_export['date'] = df_export.index
            df_export = df_export.reset_index(drop=True)
        
        # Reorder columns to put identifiers first
        id_cols = ['date', 'location', 'longitude', 'latitude']
        existing_id_cols = [col for col in id_cols if col in df_export.columns]
        data_cols = [col for col in df_export.columns if col not in existing_id_cols]
        df_export = df_export[existing_id_cols + data_cols]
        
        # Save to CSV
        df_export.to_csv(filename, index=False)
        print(f" Data saved to {filename}")
        print(f"Shape: {df_export.shape}")
        
        return filename

def main():
    """Main function to fetch data and export to okba.csv"""
    print("NASA POWER Solar Data Fetcher for Algerian Sahara")
    print("=" * 60)
    
    # Initialize the fetcher
    fetcher = NASAPowerSolarFetcher()
    
    # Create ML-ready dataset with all locations
    print("Creating comprehensive ML-ready dataset...")
    ml_dataset = fetcher.create_ml_ready_dataset(year=2020, include_derived_features=True)
    
    if ml_dataset is not None:
        print(f"\nSaving..")
        fetcher.save_to_csv(ml_dataset, filename="okba.csv")
        
        # Display summary statistics
        print(f"\nDataset Summary:")
        print(f"Total records: {len(ml_dataset)}")
        print(f"Locations: {', '.join(ml_dataset['location'].unique())}")
        print(f"Date range: {ml_dataset['date'].min()} to {ml_dataset['date'].max()}")
        print(f"Total features: {len(ml_dataset.columns)}")
        
        # Display key solar parameters statistics
        solar_params = ['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI']
        print(f"\n☀️ Key Solar Parameters Summary:")
        for param in solar_params:
            if param in ml_dataset.columns:
                valid_count = ml_dataset[param].notna().sum()
                if valid_count > 0:
                    mean_val = ml_dataset[param].mean()
                    print(f"{param}: {valid_count} valid values, mean = {mean_val:.2f}")
        
        print(f"\nDataset successfully exported to okba.csv")
        
    else:
        print("Failed to create dataset. Please check your internet connection and try again.")
        
        # Try with a single location as fallback
        print("\nTrying fallback approach with single location...")
        single_result, approach = fetcher.try_multiple_approaches("Tamanrasset", 2020)
        
        if single_result is not None:
            # Add location info and save
            single_result['location'] = 'Tamanrasset'
            single_result['longitude'] = fetcher.locations['Tamanrasset'][0]
            single_result['latitude'] = fetcher.locations['Tamanrasset'][1]
            single_result['date'] = single_result.index
            single_result = single_result.reset_index(drop=True)
            
            fetcher.save_to_csv(single_result, filename="okba.csv")
            print("Fallback dataset saved to okba.csv")
        else:
            print("Even fallback approach failed. Please check NASA POWER API availability.")

if __name__ == "__main__":
    main()