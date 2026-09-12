import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import zarr
from datetime import datetime
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

#Input directories and files
input_dir = "/gpfs/work4/0/dynql/GROW-Qual/"
csv_dir = os.path.join(input_dir, "gwms-csv")
site_info = os.path.join(input_dir, "gwms_site_info.csv")
units_file_csv = os.path.join(input_dir, "gwq_variable_list.csv")
ggmn_zarr_path = "/gpfs/work4/0/dynql/GROW-Qual/auxiliary/GGMN/GGMN.zarr"

#Output directories and file paths
output_dir = "/gpfs/work4/0/dynql/GROW-Qual/"
os.makedirs(output_dir, exist_ok=True)

output_zarr_dir = os.path.join(output_dir, "GROW-Qual.zarr")
output_wq_linkages_path = os.path.join(output_dir, "GROW-Qual_linkages.parquet")

#Define time range
START_DATE = pd.Timestamp('1980-01-01').date()
END_DATE = pd.Timestamp('2025-09-30').date()
WQ_UNITS = {}

#Define chunking strategy
ZARR_CHUNKS = {
    'time': 16710,
    'gwms_id': 2500,
    'ggmn_id': 500,  #aligned with GGMN.zarr
}

#parallelisation settings
N_WORKERS = min(cpu_count() - 1, 8) #max 8 workers

###---------------------------------------------------###
###                   Functions                       ###
###---------------------------------------------------###

def load_wq_units(units_file_csv):
    """Load units per water quality parameter from the variable list file"""
    df_units = pd.read_csv(units_file_csv)
    return dict(zip(df_units['variable_code'], df_units['target_unit']))

def get_wq_units(param_name):
    """Get units for a water quality parameter."""
    return WQ_UNITS.get(param_name, 'unknown')

def map_combined_flag(flag_value, imputation_method):
    """Map flag and imputation_method columns to a combined integer flag (0 = observed; 1 = LOD [direct substitution]; 2 = LOD [ROS], 3 = outlier, 255 = missing"""
    
    flag_str = str(flag_value).strip() if pd.notna(flag_value) else ''
    
    if flag_str == '*':
        return 3 #outlier
    elif flag_str == '<':
        method_str = str(imputation_method).strip().upper() if pd.notna(imputation_method) else ''
        if method_str == 'ROS':
            return 2 #ROS imputed
        else:
            return 1 #LOD/2 (default for censored)
    else:
        return 0 #observed


def load_ggmn_zarr(zarr_path):
    """Load GGMN zarr store"""
    
    print(f"Loading GGMN.zarr from {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)
    
    ggmn_ids = sorted(str(gid) for gid in ds.ggmn_id.values)
    gwl_vars = [v for v in ds.data_vars if v not in ('ggmn_lat', 'ggmn_lon')]
    ggmn_id_to_idx = {str(gid): i for i, gid in enumerate(ds.ggmn_id.values)}
    
    ggmn_lats = ds['ggmn_lat'].sel(ggmn_id=ggmn_ids).values.astype('f4')
    ggmn_lons = ds['ggmn_lon'].sel(ggmn_id=ggmn_ids).values.astype('f4')
    
    print(f"  Loaded {len(ggmn_ids)} ggmn stations from GGMN.zarr")
    print(f"  {len(gwl_vars)} GWL parameter(s): {gwl_vars}")
    
    return ds, ggmn_id_to_idx, ggmn_ids, ggmn_lats, ggmn_lons, gwl_vars


def load_linkages_and_gwms_ids(site_info_csv, sites_to_process=None):
    """Load gwms_id and linkages (e.g. ggmn_id, grow_id, distances, WHYMAP)"""
    
    df_linkages = pd.read_csv(site_info_csv, dtype={"gwms_id": str, "ggmn_id": str, "grow_id": str, "grow_id_from_ggmn": str})
    
    #get unique gwms_ids
    if sites_to_process is not None:
        gwms_ids = df_linkages[df_linkages['gwms_id'].isin(sites_to_process)]['gwms_id'].unique()
    else:
        gwms_ids = df_linkages['gwms_id'].unique()
    
    gwms_ids_sorted = sorted([str(gid) for gid in gwms_ids])
    
    print(f"Linkages for {len(gwms_ids_sorted)} gwms_id loaded from {site_info_csv}")
    return df_linkages, gwms_ids_sorted


def get_param_names(csv_dir):
    """Get parameter names from the water quality .csv directory"""
    
    param_files = [f for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
    return sorted([os.path.splitext(f)[0] for f in param_files])


def get_wq_gwms_ids(csv_dir, wq_params, start_date, end_date):
    """Scan the gwms_id and dates columns of every groundwater quality csv to only select gwms_id's that have data in the zarr time-frame"""
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    ids_with_data = set()
    for param_name in wq_params:
        csv_path = os.path.join(csv_dir, f"{param_name}.csv")
        df = pd.read_csv(csv_path, usecols=["gwms_id", "dates"], dtype={"gwms_id": str})
        dates = pd.to_datetime(df["dates"], errors="coerce")
        in_range = dates.between(start_ts, end_ts)
        ids_with_data.update(df.loc[in_range, "gwms_id"].dropna().unique())
    print(f"  {len(ids_with_data):,} gwms_id have >=1 WQ observation within the zarr's time range")
    return ids_with_data


###---------------------------------------------------###
###                Initialize Zarr Store              ###
###---------------------------------------------------###

def initialize_zarr_store(output_zarr_dir, gwms_ids, ggmn_ids, ggmn_lats, ggmn_lons, dates_sorted, gwl_vars, wq_params, df_linkages, ds_ggmn):
    """initialise .Zarr with groundwater depths and levels (along ggmn_id dimension) and water quality (along gwms_id dimension)"""
    
    print(f"Initialising Zarr at {output_zarr_dir}")
    
    if os.path.exists(output_zarr_dir):
        print(f"  Clearing existing store...")
        shutil.rmtree(output_zarr_dir)
    
    time_coord = pd.to_datetime(dates_sorted)
    n_time = len(dates_sorted)
    n_gwms = len(gwms_ids)
    n_ggmn = len(ggmn_ids)
    
    #create groundwater quantity arrays
    print(f"  Writing {len(gwl_vars)} groundwater quantity array(s) with ggmn_id coordinates...")
    ggmn_coord = np.array(ggmn_ids, dtype='U50')
    
    gwl_data = {}
    for var in gwl_vars:
        gwl_data[var] = xr.DataArray(
            np.full((n_ggmn, n_time), np.nan, dtype='f4'),
            dims=['ggmn_id', 'time'],
            coords={'ggmn_id': ggmn_coord, 'time': time_coord},
            attrs=dict(ds_ggmn[var].attrs)
        )
    gwl_data['ggmn_lat'] = xr.DataArray(
        ggmn_lats, dims=['ggmn_id'], coords={'ggmn_id': ggmn_coord},
        attrs={'units': 'degrees_north', 'long_name': 'GGMN station latitude'})
    gwl_data['ggmn_lon'] = xr.DataArray(
        ggmn_lons, dims=['ggmn_id'], coords={'ggmn_id': ggmn_coord},
        attrs={'units': 'degrees_east', 'long_name': 'GGMN station longitude'})
    
    ds_gwl = xr.Dataset(gwl_data)
    encoding = {var: {'chunks': (ZARR_CHUNKS['ggmn_id'], ZARR_CHUNKS['time'])} for var in gwl_vars}
    encoding['ggmn_lat'] = {'chunks': (ZARR_CHUNKS['ggmn_id'],)}
    encoding['ggmn_lon'] = {'chunks': (ZARR_CHUNKS['ggmn_id'],)}
    ds_gwl.to_zarr(output_zarr_dir, mode='w', encoding=encoding, consolidated=False, zarr_format=2)
    del ds_gwl
    
    #prepare gwms lat/lon arrays (aligned with gwms_ids)
    gwms_coord = np.array(gwms_ids, dtype='U50')
    gwms_meta_dict = df_linkages.set_index("gwms_id").to_dict("index")
    gwms_lats = np.array([gwms_meta_dict.get(gid, {}).get('gwms_lat', np.nan) for gid in gwms_ids], dtype='f4')
    gwms_lons = np.array([gwms_meta_dict.get(gid, {}).get('gwms_lon', np.nan) for gid in gwms_ids], dtype='f4')
    gwms_countries = np.array([gwms_meta_dict.get(gid, {}).get('country_name', '') for gid in gwms_ids], dtype='U100')
    gwms_hydrobasins = np.array([gwms_meta_dict.get(gid, {}).get('hydrobasin_level12', '') for gid in gwms_ids], dtype='U100')
    
    #create water quality arrays (obs, flag, and detection limit)
    print(f"  Writing {len(wq_params)} water quality parameter arrays (with flag and detection limit)...")
    
    for idx, param_name in enumerate(wq_params):
        print(f"    - {param_name} ({idx+1}/{len(wq_params)})")
        
        units = get_wq_units(param_name)
        
        ds_wq = xr.Dataset({
            param_name: xr.DataArray(
                np.full((n_gwms, n_time), np.nan, dtype='f4'),
                dims=['gwms_id', 'time'],
                coords={'gwms_id': gwms_coord, 'time': time_coord},
                attrs={'units': units, 'long_name': param_name}
            ),
            f'{param_name}_flag': xr.DataArray(
                np.full((n_gwms, n_time), 255, dtype='u1'),
                dims=['gwms_id', 'time'],
                coords={'gwms_id': gwms_coord, 'time': time_coord},
                attrs={
                    'long_name': f'{param_name} observation flag',
                    'flag_values': '0, 1, 2, 3, 255',
                    'flag_meanings': '0=observed, 1=below_detection_LOD2, 2=below_detection_ROS, 3=outlier_flagged, 255=missing'
                }
            ),
            f'{param_name}_detection_limit': xr.DataArray(
                np.full((n_gwms, n_time), np.nan, dtype='f4'),
                dims=['gwms_id', 'time'],
                coords={'gwms_id': gwms_coord, 'time': time_coord},
                attrs={'units': units, 'long_name': f'{param_name} detection limit'}
            )
        })
        
        #add gwms coordinates on first parameter
        if idx == 0:
            ds_wq['gwms_lat'] = xr.DataArray(
                gwms_lats, dims=['gwms_id'], coords={'gwms_id': gwms_coord},
                attrs={'units': 'degrees_north', 'long_name': 'gwms station latitude'})
            ds_wq['gwms_lon'] = xr.DataArray(
                gwms_lons, dims=['gwms_id'], coords={'gwms_id': gwms_coord},
                attrs={'units': 'degrees_east', 'long_name': 'gwms station longitude'})
            ds_wq['country_name'] = xr.DataArray(
                gwms_countries, dims=['gwms_id'], coords={'gwms_id': gwms_coord},
                attrs={'long_name': 'Country name'})
            ds_wq['hydrobasin_level12'] = xr.DataArray(
                gwms_hydrobasins, dims=['gwms_id'], coords={'gwms_id': gwms_coord},
                attrs={'long_name': 'HydroBasin Level 12 ID'})
            encoding = {
                param_name: {'chunks': (ZARR_CHUNKS['gwms_id'], ZARR_CHUNKS['time'])},
                f'{param_name}_flag': {'chunks': (ZARR_CHUNKS['gwms_id'], ZARR_CHUNKS['time'])},
                f'{param_name}_detection_limit': {'chunks': (ZARR_CHUNKS['gwms_id'], ZARR_CHUNKS['time'])},
                'gwms_lat': {'chunks': (ZARR_CHUNKS['gwms_id'],)},
                'gwms_lon': {'chunks': (ZARR_CHUNKS['gwms_id'],)},
                'country_name': {'chunks': (ZARR_CHUNKS['gwms_id'],)},
                'hydrobasin_level12': {'chunks': (ZARR_CHUNKS['gwms_id'],)}
            }
        else:
            encoding = {
                param_name: {'chunks': (ZARR_CHUNKS['gwms_id'], ZARR_CHUNKS['time'])},
                f'{param_name}_flag': {'chunks': (ZARR_CHUNKS['gwms_id'], ZARR_CHUNKS['time'])},
                f'{param_name}_detection_limit': {'chunks': (ZARR_CHUNKS['gwms_id'], ZARR_CHUNKS['time'])}
            }
        
        ds_wq.to_zarr(output_zarr_dir, mode='a', encoding=encoding, consolidated=False, zarr_format=2)
        del ds_wq
    
    print(f"  Zarr store initialised:")
    print(f"    {n_ggmn} ggmn stations (copied directly from GGMN.zarr)")
    print(f"    {n_gwms} gwms stations")
    print(f"    {n_time} time steps")
    print(f"    {len(gwl_vars)} GWL parameters")
    print(f"    {len(wq_params)} water quality parameters (each with flag and detection limit)")
    print()

###---------------------------------------------------###
###                 Process Data                      ###
###---------------------------------------------------###

def process_gwl_data(output_zarr_dir, ggmn_ids, ds_ggmn, gwl_vars, dates):
    """Copy groundwater level and depth data from GGMN.zarr into the combined zarr"""
    
    print(f"Processing GWL data for {len(ggmn_ids)} ggmn stations, {len(gwl_vars)} parameter(s)...")
    
    ds_aligned = ds_ggmn.reindex(ggmn_id=ggmn_ids).reindex(time=pd.to_datetime(dates))
    z = zarr.open_group(output_zarr_dir, mode='r+')
    
    for var in gwl_vars:
        values = ds_aligned[var].values.astype('f4')
        z[var][:, :] = values
        processed = int(np.isfinite(values).any(axis=1).sum())
        pct = (processed / len(ggmn_ids)) * 100 if len(ggmn_ids) > 0 else 0
        print(f"  {var}: {processed:,} / {len(ggmn_ids):,} stations ({pct:.1f}%)")
    
    print(f"Finished processing groundwater quantity data")


def _aggregate_daily(df_site):
    """Aggregate multiple same-day observations for a single station to one row per day."""
    
    result_rows = []
    for date_val, day_df in df_site.groupby('dates'):
        
        #aggregate obs
        obs_vals = day_df['obs'].dropna()
        obs_agg = obs_vals.mean() if len(obs_vals) > 0 else np.nan
        
        #aggregate flag with priority: * > < > ""
        flags = day_df['flag'].fillna('').astype(str).str.strip() if 'flag' in day_df.columns else pd.Series([''] * len(day_df))
        if (flags == '*').any():
            flag_agg = '*'
        elif (flags == '<').any():
            flag_agg = '<'
        else:
            flag_agg = ''
        
        #aggregate imputation_method: first non-empty
        if 'imputation_method' in day_df.columns:
            methods = day_df['imputation_method'].fillna('').astype(str).str.strip()
            non_empty = methods[methods != '']
            method_agg = non_empty.iloc[0] if len(non_empty) > 0 else ''
        else:
            method_agg = ''
        
        #aggregate detection_limit: mean of non-NaN
        if 'detection_limit' in day_df.columns:
            dl_vals = pd.to_numeric(day_df['detection_limit'], errors='coerce').dropna()
            dl_agg = dl_vals.mean() if len(dl_vals) > 0 else np.nan
        else:
            dl_agg = np.nan
        
        result_rows.append({
            'dates': date_val,
            'obs': obs_agg,
            'flag': flag_agg,
            'imputation_method': method_agg,
            'detection_limit': dl_agg
        })
    
    return pd.DataFrame(result_rows)


def process_single_wq_parameter(param_name, output_zarr_dir, gwms_ids, csv_dir, start_date, n_time, chunk_size, sites_to_process=None):
    """process a single water quality parameter (with flag and detection limit). Aggregate multiple observations for the same gwms_id and date to a single daily value"""
    
    csv_path = os.path.join(csv_dir, f"{param_name}.csv")
    
    try:
        df_param = pd.read_csv(csv_path, parse_dates=["dates"], date_format="%Y-%m-%d", low_memory=False)
        df_param["dates"] = pd.to_datetime(df_param["dates"], errors="coerce")
        
        #normalise column names: support both 'flag' and legacy 'limit_flag'
        if 'limit_flag' in df_param.columns and 'flag' not in df_param.columns:
            df_param = df_param.rename(columns={'limit_flag': 'flag'})
        
        #ensure expected columns exist
        for col in ('flag', 'imputation_method', 'detection_limit'):
            if col not in df_param.columns:
                df_param[col] = np.nan if col == 'detection_limit' else ''
        
        df_param['flag'] = df_param['flag'].fillna('').astype(str).str.strip()
        df_param['imputation_method'] = df_param['imputation_method'].fillna('').astype(str).str.strip()
        df_param['detection_limit'] = pd.to_numeric(df_param['detection_limit'], errors='coerce')
        
        if sites_to_process is not None:
            df_param = df_param[df_param["gwms_id"].isin(sites_to_process)]
        
        z = zarr.open_group(output_zarr_dir, mode='r+')
        wq_array = z[param_name]
        flag_array = z[f'{param_name}_flag']
        dl_array = z[f'{param_name}_detection_limit']
        
        grouped = df_param.groupby("gwms_id")
        
        n_chunks = (len(gwms_ids) + chunk_size - 1) // chunk_size
        stations_processed = 0
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(gwms_ids))
            chunk_gwms = gwms_ids[start_idx:end_idx]
            
            chunk_data  = np.full((len(chunk_gwms), n_time), np.nan, dtype='f4')
            chunk_flags = np.full((len(chunk_gwms), n_time), 255, dtype='u1')  # 255 = missing
            chunk_dl    = np.full((len(chunk_gwms), n_time), np.nan, dtype='f4')
            
            for i, gwms_id in enumerate(chunk_gwms):
                if gwms_id not in grouped.groups:
                    continue
                
                df_site = grouped.get_group(gwms_id)
                
                #aggregate to one value per day
                df_daily = _aggregate_daily(df_site)
                
                for _, row in df_daily.iterrows():
                    days_from_start = (row["dates"].date() - start_date).days
                    if not (0 <= days_from_start < n_time):
                        continue
                    
                    obs_value = row["obs"]
                    if pd.notna(obs_value):
                        chunk_data[i, days_from_start]  = obs_value
                        chunk_flags[i, days_from_start] = map_combined_flag(
                            row["flag"], row["imputation_method"]
                        )
                    
                    dl_value = row["detection_limit"]
                    if pd.notna(dl_value):
                        chunk_dl[i, days_from_start] = dl_value
                
                if not df_site.empty:
                    stations_processed += 1
            
            wq_array[start_idx:end_idx, :] = chunk_data
            flag_array[start_idx:end_idx, :] = chunk_flags
            dl_array[start_idx:end_idx, :] = chunk_dl
        
        return (param_name, stations_processed, True, None)
    
    except Exception as e:
        return (param_name, 0, False, str(e))


def process_wq_data(output_zarr_dir, gwms_ids, csv_dir, wq_params, start_date, n_time, sites_to_process=None):
    """process water quality data for .Zarr in parallel"""
    
    print(f"Processing water quality data (with flag and detection limit) for {len(wq_params)} parameters using {N_WORKERS} workers...")
    
    chunk_size = ZARR_CHUNKS['gwms_id']
    
    process_func = partial(process_single_wq_parameter,
                          output_zarr_dir=output_zarr_dir,
                          gwms_ids=gwms_ids,
                          csv_dir=csv_dir,
                          start_date=start_date,
                          n_time=n_time,
                          chunk_size=chunk_size,
                          sites_to_process=sites_to_process)
    
    processed = 0
    total_stations = 0
    
    with Pool(processes=N_WORKERS) as pool:
        for idx, (param_name, stations_processed, success, error) in enumerate(pool.imap(process_func, wq_params)):
            pct = ((idx + 1) / len(wq_params)) * 100
            
            if success:
                print(f"  [{idx+1:3d}/{len(wq_params)}] {pct:5.1f}% | {param_name:20s} | {stations_processed:4d} stations")
                processed += 1
                total_stations += stations_processed
            else:
                print(f"  [{idx+1:3d}/{len(wq_params)}] {pct:5.1f}% | {param_name:20s} | ERROR: {error}")
    
    pct = (processed / len(wq_params)) * 100 if len(wq_params) > 0 else 0
    print(f"Finished processing water quality data ({pct:.1f}% success rate, {total_stations} total stations)")


###---------------------------------------------------------###
###              Detailed metadata for each station         ###
###---------------------------------------------------------###

def _idx_to_date_str(idx_series, dates_index, n_time, sentinel_is_min):
    idx_values = idx_series.to_numpy()
    valid = (idx_values < n_time) if sentinel_is_min else (idx_values >= 0)
    safe_idx = np.clip(idx_values, 0, n_time - 1)
    picked = dates_index[safe_idx].strftime('%Y-%m-%d')
    return pd.Series(np.where(valid, picked, None), index=idx_series.index)


def add_observation_metadata_to_linkages(output_zarr_dir, df_linkages, gwms_ids):
    """Add comprehensive observation metadata to the linkages dataframe"""

    print("Calculating observation metadata for each gwms station (dask, chunked, lazy)...")

    ds = xr.open_zarr(output_zarr_dir, consolidated=False)
    ds = ds.chunk({'gwms_id': ZARR_CHUNKS['gwms_id']})  #explicit, don't rely on auto-chunking

    #water quality variables only (indexed by gwms_id & time)
    wq_vars = [v for v in ds.data_vars if not v.endswith('_flag') and not v.endswith('_detection_limit') and ds[v].dims == ('gwms_id', 'time')]

    #water quantity variables (indexed by ggmn_id & time)
    gwl_vars = [v for v in ds.data_vars if ds[v].dims == ('ggmn_id', 'time')]

    gwms_index = pd.Index(gwms_ids, name='gwms_id')
    time_da = ds['time']
    n_time = ds.sizes['time']
    dates_index = pd.DatetimeIndex(pd.to_datetime(time_da.values))
    day_idx = xr.DataArray(np.arange(n_time), dims=['time'], coords={'time': time_da})

    total_obs = pd.Series(0, index=gwms_index)
    parameters_measured = pd.Series(0, index=gwms_index)
    any_obs_mask = None
    param_cols = {}

    for idx, var in enumerate(wq_vars, 1):
        print(f"  [{idx}/{len(wq_vars)}] {var}")

        val_da = ds[var]
        flag_da = ds[f'{var}_flag']
        mask = val_da.notnull()

        obs_count = mask.sum(dim='time').to_pandas().reindex(gwms_index)
        param_cols[f'obs_{var}'] = obs_count
        param_cols[f'obs_{var}_observed'] = ((flag_da == 0) & mask).sum(dim='time').to_pandas().reindex(gwms_index)
        param_cols[f'obs_{var}_LOD2'] = ((flag_da == 1) & mask).sum(dim='time').to_pandas().reindex(gwms_index)
        param_cols[f'obs_{var}_ROS'] = ((flag_da == 2) & mask).sum(dim='time').to_pandas().reindex(gwms_index)
        param_cols[f'obs_{var}_outlier'] = ((flag_da == 3) & mask).sum(dim='time').to_pandas().reindex(gwms_index)

        #get first/last observation date via integer day-index min/max
        first_idx = day_idx.where(mask, other=n_time).min(dim='time').to_pandas().reindex(gwms_index)
        last_idx  = day_idx.where(mask, other=-1).max(dim='time').to_pandas().reindex(gwms_index)
        param_cols[f'start_{var}'] = _idx_to_date_str(first_idx, dates_index, n_time, sentinel_is_min=True)
        param_cols[f'end_{var}']   = _idx_to_date_str(last_idx, dates_index, n_time, sentinel_is_min=False)

        total_obs = total_obs.add(obs_count, fill_value=0)
        parameters_measured = parameters_measured.add((obs_count > 0).astype(int), fill_value=0)
        any_obs_mask = mask if any_obs_mask is None else (any_obs_mask | mask) 

    print("  Computing overall start/end/observation_years/most_observed_parameter...")

    overall_first_idx = day_idx.where(any_obs_mask, other=n_time).min(dim='time').to_pandas().reindex(gwms_index)
    overall_last_idx = day_idx.where(any_obs_mask, other=-1).max(dim='time').to_pandas().reindex(gwms_index)
    start_date_str = _idx_to_date_str(overall_first_idx, dates_index, n_time, sentinel_is_min=True)
    end_date_str = _idx_to_date_str(overall_last_idx, dates_index, n_time, sentinel_is_min=False)

    #distinct years with >=1 observation, any parameter
    years_with_obs = any_obs_mask.groupby('time.year').any(dim='time')
    observation_years = years_with_obs.sum(dim='year').to_pandas().reindex(gwms_index)

    #identify which parameter has the most observations, per station
    obs_counts_df = pd.DataFrame({var: param_cols[f'obs_{var}'] for var in wq_vars})
    most_observed_parameter = obs_counts_df.idxmax(axis=1).where(obs_counts_df.sum(axis=1) > 0)

    df_linkages = df_linkages.set_index('gwms_id')
    df_linkages['total_observations'] = total_obs
    df_linkages['start_date'] = start_date_str
    df_linkages['end_date'] = end_date_str
    df_linkages['observation_years'] = observation_years
    df_linkages['parameters_measured'] = parameters_measured
    df_linkages['most_observed_parameter'] = most_observed_parameter

    
    param_df = pd.DataFrame(param_cols, index=gwms_index)
    df_linkages = pd.concat([df_linkages, param_df], axis=1)

    #groundwater quantity observation counts, looked up via the ggmn_id link column
    if gwl_vars:
        for var in gwl_vars:
            counts = ds[var].notnull().sum(dim='time').to_pandas()
            df_linkages[f'obs_gwl_{var}'] = df_linkages['ggmn_id'].map(counts.to_dict())

    df_linkages['has_gwl'] = df_linkages['ggmn_id'].notna()

    return df_linkages.reset_index()


###---------------------------------------------------###
###       Linking gwms_id, ggmn_id and grow_id        ###
###---------------------------------------------------###

def create_linkages(df_linkages, output_wq_path, sites_to_process=None):
    print(f"Creating parquet file with metadata...")
    
    if sites_to_process is None:
        df_wq = df_linkages.copy()
    else:
        df_wq = df_linkages[df_linkages["gwms_id"].isin(sites_to_process)].copy()
    
    for col in df_wq.select_dtypes(include="object").columns:
        df_wq[col] = df_wq[col].astype("string")
    
    df_wq.to_parquet(output_wq_path, index=False, compression="snappy")
    print(f"  wrote {len(df_wq):,} rows to {output_wq_path}")


###---------------------------------------------------###
###                    Runner                         ###
###---------------------------------------------------###

if __name__ == "__main__":
    
    print(f"Creating {output_zarr_dir}...")
    
    #load units from CSV file
    WQ_UNITS = load_wq_units(units_file_csv)
    
    #load groundwater quantity data (from GGMN.zarr)
    ds_ggmn, ggmn_id_to_idx, ggmn_ids, ggmn_lats, ggmn_lons, gwl_vars = load_ggmn_zarr(ggmn_zarr_path)
    
    #find which gwms_id have >=1 observations (in the specified time range) for writing to .zarr
    wq_params = get_param_names(csv_dir)
    sites_to_process = get_wq_gwms_ids(csv_dir, wq_params, START_DATE, END_DATE)
    
    #load linkages and gwms_ids
    df_linkages, gwms_ids = load_linkages_and_gwms_ids(site_info, sites_to_process=sites_to_process)
    
    #create date range
    dates_sorted = pd.date_range(start=START_DATE, end=END_DATE, freq='D').date.tolist()
    n_time = len(dates_sorted)
    
    #initialize zarr store with coordinates
    initialize_zarr_store(output_zarr_dir, gwms_ids, ggmn_ids, ggmn_lats, ggmn_lons, dates_sorted, gwl_vars, wq_params, df_linkages, ds_ggmn)
    
    #process groundwater quantity from GGMN.zarr, process water quality data directly from csvs
    process_gwl_data(output_zarr_dir, ggmn_ids, ds_ggmn, gwl_vars, dates_sorted)
    process_wq_data(output_zarr_dir, gwms_ids, csv_dir, wq_params, START_DATE, n_time, sites_to_process=sites_to_process)
    ds_ggmn.close()
    
    #process linkages parquet
    df_linkages = add_observation_metadata_to_linkages(output_zarr_dir, df_linkages, gwms_ids)
    create_linkages(df_linkages, output_wq_linkages_path, sites_to_process=sites_to_process)
    
    #consolidate metadata
    zarr.consolidate_metadata(output_zarr_dir)
    
    print("Processing complete!")
