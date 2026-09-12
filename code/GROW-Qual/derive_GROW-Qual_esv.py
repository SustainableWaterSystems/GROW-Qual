# -*- coding: utf-8 -*-
"""
Derive GROW earth-system variables for groundwater quality stations (i.e. gwms_id)
- python3 derive_GROW-Qual_esv.py <step>
- python3 derive_GROW-Qual_esv.py --all
Dependencies: pandas, numpy, geopandas, shapely, xarray, dask, netCDF4,rioxarray, pyproj, pyarrow.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

###---------------------------------------------------###
###                     Setup                          ###
###---------------------------------------------------###

#directories for GROW earth system variables
INPUT_ROOT = "/gpfs/work4/0/dynql/GROW-Qual/auxiliary/input_esv/"
OUTPUT_ROOT = "/gpfs/work4/0/dynql/GROW-Qual/auxiliary/output_esv/"

#determine stations for processing (directly from the .zarr)
LINKAGES_PARQUET = "/gpfs/work4/0/dynql/GROW-Qual/GROW-Qual_linkages.parquet"
STATION_ID_COL, LAT_COL, LON_COL = "gwms_id", "gwms_lat", "gwms_lon"
FIRST_OBS_COL, LAST_OBS_COL = "first_observation_date", "last_observation_date"
TOTAL_OBS_COL = "total_observations"

#map linkages parquet
_LINKAGES_COLS = {
    "gwms_id": STATION_ID_COL,
    "gwms_lat": LAT_COL,
    "gwms_lon": LON_COL,
    "start_date": FIRST_OBS_COL,
    "end_date": LAST_OBS_COL,
    "total_observations": TOTAL_OBS_COL,
}

#start and end time for deriving earth system variables (align with .zarr)
GLOBAL_START = pd.Timestamp("1980-01-01")
GLOBAL_END = pd.Timestamp("2025-12-31")

SKIP_EXISTING_STATIONS = True #if false, process only new stations


###---------------------------------------------------###
###         Data source registry (edit paths here)     ###
###---------------------------------------------------###

ROOT = INPUT_ROOT
PATHS = {    
    #geosphere
    "dem": ROOT + "geosphere/topography/dem.tif",
    "slope": ROOT + "geosphere/topography/slope.tif",
    "glim": ROOT + "geosphere/GLiM/glim.tif",
    "glim_codes": ROOT + "geosphere/GLiM/glim_codes.csv", # sep=";"
    "whymap_karst": ROOT + "geosphere/WHYMAP/whymap_karst.gpkg",
    "permeability": ROOT + "geosphere/GLYHMPS2.0/permeability.gpkg",
    "porosity": ROOT + "geosphere/GLHYMPS/porosity.gpkg",
    "soil_texture_topsoil": ROOT + "geosphere/HiHydroKlass/soil_texture_topsoil.tif",
    "soil_texture_subsoil": ROOT + "geosphere/HiHydroKlass/soil_texture_subsoil.tif",
    "soil_texture_codes": ROOT + "geosphere/HiHydroKlass/soil_texture_codes.csv", # sep=";"
    "soil_ksat_topsoil": ROOT + "geosphere/HiHydroKlass/soil_ksat_topsoil.tif",
    "soil_ksat_subsoil": ROOT + "geosphere/HiHydroKlass/soil_ksat_subsoil.tif",

    #hydrosphere
    "dist_streams": ROOT + "hydrosphere/Distance_perennial_streams/distance_to_streams.tif",
    "drainage_density": ROOT + "hydrosphere/Drainage_Density_ESRI_54012/drainage_density.gpkg",

    #hydroatlas
    "glacier_permafrost": ROOT + "hydroatlas/BasinATLAS/basinatlas_lev12.gpkg",

    #biosphere
    "ggde": ROOT + "biosphere/GGDE/ggde.tif",
    "ggde_codes": ROOT + "biosphere/GGDE/ggde_codes.csv", # sep=";"
    "ndvi": ROOT + "biosphere/NDVI/ndvi.nc",

    #anthroposphere
    "gw_scapes": ROOT + "anthroposphere/Groundwaterscapes/groundwaterscapes.tif",
    "abstract_ind": ROOT + "anthroposphere/ISIMIP_withdrawals/withdrawal_industrial.nc",
    "abstract_dom": ROOT + "anthroposphere/ISIMIP_withdrawals/withdrawal_domestic.nc",
    "lu_totals": ROOT + "anthroposphere/Land_use/land_use_totals.nc",
    "lu_urban": ROOT + "anthroposphere/Land_use/urban_area.nc",

    #atmosphere
    "koeppen_geiger": ROOT + "atmosphere/Climate_zones/koeppen_geiger.tif",
    "koeppen_geiger_codes": ROOT + "atmosphere/Climate_zones/koeppen_geiger_codes.csv",  # sep=";"
    "hydrobelts": ROOT + "atmosphere/Hydroregions/hydrobelts.gpkg",
    "hydrobelts_codes": ROOT + "atmosphere/Hydroregions/hydrobelts_codes.csv",  # sep="\t"
    "gpcc": ROOT + "atmosphere/GPCC/precipitation_gpcc.nc",
    "mswep": ROOT + "atmosphere/MSWEP/precipitation_mswep.nc",
    "era5_pet": ROOT + "atmosphere/ERA5-Land/pet.nc",
    "era5_temp": ROOT + "atmosphere/ERA5-Land/temperature.nc",
    "era5_snow": ROOT + "atmosphere/ERA5-Land/snow_depth.nc",
    "era5_lai_low": ROOT + "atmosphere/ERA5-Land/lai_low.nc",
    "era5_lai_high": ROOT + "atmosphere/ERA5-Land/lai_high.nc",
    "gleam_pet": ROOT + "atmosphere/GLEAM/potential_evapotranspiration.nc",
    "gleam_aet": ROOT + "atmosphere/GLEAM/actual_evapotranspiration.nc",
    "gleam_interception": ROOT + "atmosphere/GLEAM/interception.nc",
}


###---------------------------------------------------###
###          Station loading & time windows            ###
###---------------------------------------------------###

def load_stations():
    """Load stations from the linkages parquet, with point geometries attached."""
    df = pd.read_parquet(LINKAGES_PARQUET, columns=list(_LINKAGES_COLS.keys()))
    df = df.rename(columns=_LINKAGES_COLS)[list(_LINKAGES_COLS.values())]
    df[FIRST_OBS_COL] = pd.to_datetime(df[FIRST_OBS_COL])
    df[LAST_OBS_COL] = pd.to_datetime(df[LAST_OBS_COL])
    stations = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]),
        crs="EPSG:4326",
    )
    print(f"Loaded {len(stations)} stations from {LINKAGES_PARQUET}")
    return stations


def compute_station_windows(stations):
    """Per-station window for deriving earth system variables (full calendar year before and after first and last observation, respectively, clipped to [GLOBAL_START, GLOBAL_END]. """
    windows = stations[[STATION_ID_COL]].copy()
    start_year = stations[FIRST_OBS_COL].dt.year - 1
    end_year = stations[LAST_OBS_COL].dt.year + 1
    windows["window_start"] = pd.to_datetime(start_year.astype(str) + "-01-01").clip(lower=GLOBAL_START)
    windows["window_end"] = pd.to_datetime(end_year.astype(str) + "-12-31").clip(upper=GLOBAL_END)
    return windows


###---------------------------------------------------###
###           Generic extraction functions            ###
###---------------------------------------------------###

def extract_raster_at_points(stations, rasterfile, col_name, transform=False):
    """Nearest-neighbour value of a raster at every station"""
    import rasterio

    with rasterio.open(rasterfile) as src:
        xs = stations[LON_COL].values.astype("float64")
        ys = stations[LAT_COL].values.astype("float64")
        if transform and src.crs is not None and src.crs.to_string() != "EPSG:4326":
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = transformer.transform(xs, ys)

        values = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype="float64")
        if src.nodata is not None:
            values = np.where(values == src.nodata, np.nan, values)

    return pd.Series(values, index=stations.index, name=col_name)


def extract_vector_at_points(stations, vector, columns):
    """spatial join of polygon/line attributes to station points"""
    if vector.crs != stations.crs:
        vector = vector.to_crs(stations.crs)
    joined = gpd.sjoin(stations[[STATION_ID_COL, "geometry"]], vector[columns + ["geometry"]], how="left")
    joined = joined.drop_duplicates(subset=STATION_ID_COL).set_index(stations.index) #relevant if station sites exactly on a border
    return joined[columns]


def extract_transient_variable(stations, windows, file, band, lat_dim="lat", lon_dim="lon", preprocess=None, resample_to_daily=False, mask_period=None, chunk_freq="Y"):
    """process transient variables per time chunk (calendar year)"""
    
    if not Path(file).exists():
        print(f"    {file} not found - skipping")
        return pd.DataFrame(columns=[STATION_ID_COL, "time", band])

    ds = xr.open_dataset(file, chunks={})
    if preprocess is not None:
        ds = preprocess(ds)

    union_start, union_end = windows["window_start"].min(), windows["window_end"].max()

    parts = []
    for period in pd.period_range(union_start, union_end, freq=chunk_freq):
        chunk_start, chunk_end = period.start_time, period.end_time
        overlap = (windows["window_start"] <= chunk_end) & (windows["window_end"] >= chunk_start)
        if not overlap.any():
            continue
        chunk_ids = windows.loc[overlap, STATION_ID_COL].values
        chunk_stations = stations[stations[STATION_ID_COL].isin(chunk_ids)]

        t_chunk = time.time()
        sub_ds = ds.sel(time=slice(chunk_start, chunk_end))
        lat_da = xr.DataArray(chunk_stations[LAT_COL].values, dims="station", coords={"station": chunk_stations[STATION_ID_COL].values})
        lon_da = xr.DataArray(chunk_stations[LON_COL].values, dims="station", coords={"station": chunk_stations[STATION_ID_COL].values})
        extracted = sub_ds[band].sel({lat_dim: lat_da, lon_dim: lon_da}, method="nearest").compute()

        chunk_df = extracted.to_dataframe(name=band).reset_index()
        chunk_df = chunk_df.rename(columns={"station": STATION_ID_COL})[[STATION_ID_COL, "time", band]]
        parts.append(chunk_df)
        print(f"    {period} done ({len(chunk_stations)} station(s), {time.time() - t_chunk:.1f}s)")

    if not parts:
        return pd.DataFrame(columns=[STATION_ID_COL, "time", band])
    long_df = pd.concat(parts, ignore_index=True)

    if resample_to_daily:
        long_df = (
            long_df.groupby([STATION_ID_COL, long_df["time"].dt.floor("D")])[band]
            .mean()
            .reset_index()
        )

    long_df = long_df.merge(windows, on=STATION_ID_COL, how="left")
    if mask_period is not None:
        station_start = long_df["window_start"].dt.to_period(mask_period).dt.start_time
        station_end = long_df["window_end"].dt.to_period(mask_period).dt.end_time
        keep = long_df["time"].between(station_start, station_end)
    else:
        keep = long_df["time"].between(long_df["window_start"], long_df["window_end"])
    return long_df.loc[keep, [STATION_ID_COL, "time", band]].reset_index(drop=True).rename(columns={"time": "date"})


def _era5_fix_coords(ds):
    """extra processing on ERA5-Land: rename time coordinate and wrap longitude"""
    ds = ds.rename({"valid_time": "time"})
    ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180).sortby("longitude")
    return ds


###---------------------------------------------------###
###                 Static attributes                 ###
###---------------------------------------------------###

def step_koeppen_geiger(stations):
    """Koeppen-Geiger climate classification (class)"""
    raw = extract_raster_at_points(stations, PATHS["koeppen_geiger"], "koeppen_geiger")
    codes = pd.read_csv(PATHS["koeppen_geiger_codes"], sep=";")
    df = pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "koeppen_geiger": raw.values})
    df = df.merge(codes, how="left", left_on="koeppen_geiger", right_on="value")
    return df.drop(columns=["koeppen_geiger", "value", "group", "class name"])


def step_hydrobelt(stations):
    """Hydrobelt (class) from Meybeck et al. (2013) """
    hydrobelt = gpd.read_file(PATHS["hydrobelts"])
    codes = pd.read_csv(PATHS["hydrobelts_codes"], sep="\t")
    hydrobelt = hydrobelt.merge(codes, how="left", left_on="hydrobelt", right_on="code")
    hydrobel = extract_vector_at_points(stations, hydrobelt, ["hydrobel"])["hydrobel"]
    return pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "hydrobelt_class": hydrobel.values})


def step_dem(stations):
    """DEM elevation from MERIT (m asl)"""
    elevation = extract_raster_at_points(stations, PATHS["dem"], "elevation")
    return pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "ground_elevation_m_asl": elevation.values})


def step_slope(stations):
    """Topographic slope (degrees)"""
    slope = extract_raster_at_points(stations, PATHS["slope"], "slope")
    return pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "topographic_slope_degree": slope.values / 100})


def step_glim(stations):
    """GLiM rock type (class) and aquifer type (class), with aquifer type overwritten to 'karst' wherever WHYMAP has a colocated mapped polygon"""
    rock_type = extract_raster_at_points(stations, PATHS["glim"], "rock_type")
    names = pd.read_csv(PATHS["glim_codes"], sep=";")[["Value_", "rock_type_class", "aquifer_type_class"]]
    df = pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "rock_type": rock_type.values})
    df = df.merge(names, how="left", left_on="rock_type", right_on="Value_")
    df = df.drop(columns=["Value_", "rock_type"]).rename(columns={"rock_type_class": "rock_type_0-100_m_class"})

    whymap = gpd.read_file(PATHS["whymap_karst"])  #no NA interpolation: NA means "not karst"
    karst_hit = extract_vector_at_points(stations, whymap, ["rock_type"])["rock_type"]
    df.loc[karst_hit.notna().values, "aquifer_type_class"] = "karst"
    return df


def step_permeability(stations):
    """Permeability (m2) from GLHYMPS2.0"""
    glhymps2 = gpd.read_file(PATHS["permeability"])
    perm = extract_vector_at_points(stations, glhymps2, ["logK_Ferr_"])["logK_Ferr_"]
    perm = 10 ** (perm / 100)  #rescale, described in GLHYMPS readme file
    perm[perm == 1] = np.nan  #known fill-value artifact
    return pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "permeability_0-100_m_m-2": perm.values})


def step_porosity(stations):
    """Total porosity (0-1) from GLHYMPS"""
    porosity = extract_vector_at_points(stations, gpd.read_file(PATHS["porosity"]), ["Porosity"])["Porosity"]
    return pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "total_porosity_0-100_m_fraction": porosity.values})


def step_soil_texture(stations):
    """Soil texture (class), topsoil (0-30cm) and subsoil (30-200cm) from HiHydro"""
    codes = pd.read_csv(PATHS["soil_texture_codes"], sep=";")
    df = pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values})
    for depth_key, out_col in [("soil_texture_topsoil", "soil_texture_0-30_cm_class"),("soil_texture_subsoil", "soil_texture_30-200_cm_class")]:
        raw = extract_raster_at_points(stations, PATHS[depth_key], "value")
        merged = pd.DataFrame({"value": raw.values}).merge(codes, how="left", on="value")
        df[out_col] = merged["soil_texture_class"].values
    return df


def step_soil_ksat(stations):
    """Saturated hydraulic conductivity (cm/day) of topsoil and subsoil from HiHydroSoil"""
    df = pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values})
    for depth_key, out_col in [("soil_ksat_topsoil", "soil_saturated_conductivity_0-30_cm_cm_d-1"),("soil_ksat_subsoil", "soil_saturated_conductivity_30-200_cm_cm_d-1")]:
        raw = extract_raster_at_points(stations, PATHS[depth_key], "value")
        df[out_col] = raw.values * 0.0001  #rescale (scaling factor in original data)
    return df


def step_dist_streams(stations):
    """Distance to nearest perennial stream (m)"""
    dist = extract_raster_at_points(stations, PATHS["dist_streams"], "distance")
    return pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "distance_perennial_streams_m": dist.values})


def step_drainage_density(stations):
    """Drainage density (1/m)"""
    drainage = extract_vector_at_points(stations, gpd.read_file(PATHS["drainage_density"]), ["Drainage_d"])["Drainage_d"]
    return pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "drainage_density_m-1": drainage.values})


def step_glacier_permafrost(stations):
    """Glacier and permafrost cover fraction (0-1), from BasinATLAS level 12."""
    basins = gpd.read_file(PATHS["glacier_permafrost"])
    cover = extract_vector_at_points(stations, basins, ["gla_pc_use", "prm_pc_use"])
    return pd.DataFrame({
        STATION_ID_COL: stations[STATION_ID_COL].values,
        "glacier_cover_fraction": cover["gla_pc_use"].values / 100,
        "permafrost_cover_fraction": cover["prm_pc_use"].values / 100,
    })


def step_ggde(stations):
    """Groundwater dependent ecosystem (GDE) classification"""
    gde = extract_raster_at_points(stations, PATHS["ggde"], "gde").astype("Int64")
    names = pd.read_csv(PATHS["ggde_codes"], sep=";")
    merged = pd.DataFrame({"gde": gde.values}).merge(names, how="left", left_on="gde", right_on="code")
    return pd.DataFrame({
        STATION_ID_COL: stations[STATION_ID_COL].values,
        "groundwater_dependent_ecosystems_class": merged["name_"].fillna("No GDE").values,
    })


def step_groundwaterscapes(stations):
    """Groundwaterscapes ID (class)"""
    gw_id = extract_raster_at_points(stations, PATHS["gw_scapes"], "gw_id").fillna(0).astype(int)
    gw_id = gw_id.replace(0, np.nan)  #0 is a fill value created by the fillna above, not a real class
    return pd.DataFrame({STATION_ID_COL: stations[STATION_ID_COL].values, "groundwaterscapes_ID_class": gw_id.values})


###---------------------------------------------------###
###              Time-series attributes               ###
###---------------------------------------------------###

def step_mswep(stations, windows):
    """Precipitation from MSWEP, 3-hourly source resampled to daily (mm/yr-eq)"""
    df = extract_transient_variable(stations, windows, PATHS["mswep"], band="precipitation", resample_to_daily=True)
    df["precipitation_mswep_mm_year-1"] = df.pop("precipitation") * 365
    return df


def step_ndvi(stations, windows):
    """daily NDVI (ratio)"""
    df = extract_transient_variable(stations, windows, PATHS["ndvi"], band="NDVI", lat_dim="latitude", lon_dim="longitude")
    df["ndvi_ratio"] = df.pop("NDVI").where(lambda s: s.between(-1, 1)) #set values outside valid range to NaN
    return df


def step_gleam_potential_et(stations, windows):
    """Daily potential evapotranspiration (mm/yr-eq) from GLEAM"""
    df = extract_transient_variable(stations, windows, PATHS["gleam_pet"], band="Ep")
    df["potential_evapotranspiration_gleam_mm_year-1"] = df.pop("Ep") * 365
    return df


def step_gleam_actual_et(stations, windows):
    """Daily actual evapotranspiration (mm/yr-eq) from GLEAM"""
    df = extract_transient_variable(stations, windows, PATHS["gleam_aet"], band="E")
    df["actual_evapotranspiration_mm_year-1"] = df.pop("E") * 365
    return df


def step_gleam_interception(stations, windows):
    """Daily interception (mm/yr-eq) from GLEAM"""
    df = extract_transient_variable(stations, windows, PATHS["gleam_interception"], band="Ei")
    df["interception_mm_year-1"] = df.pop("Ei") * 365
    return df


def step_era5_pet(stations, windows):
    """Daily potential evapotranspiration (mm/yr-eq) from ERA5-Land"""
    df = extract_transient_variable(stations, windows, PATHS["era5_pet"], band="pev", lat_dim="latitude", lon_dim="longitude", preprocess=_era5_fix_coords)
    df["potential_evapotranspiration_era5_mm_year-1"] = df.pop("pev") * 365 * 1000 * -1
    return df


def step_era5_temperature(stations, windows):
    """Daiy air temperature (degrees C) from ERA5-Land"""
    df = extract_transient_variable(stations, windows, PATHS["era5_temp"], band="t2m", lat_dim="latitude", lon_dim="longitude", preprocess=_era5_fix_coords)
    df["air_temperature_\u00b0C"] = df.pop("t2m") - 273
    return df


def step_era5_snow_depth(stations, windows):
    """Daily snow depth [m] from ERA5-Land"""
    df = extract_transient_variable(stations, windows, PATHS["era5_snow"], band="sde", lat_dim="latitude", lon_dim="longitude", preprocess=_era5_fix_coords)
    df["snow_depth_m"] = df.pop("sde").clip(lower=0) #clip negative values to 0
    return df


def step_era5_lai_low(stations, windows):
    """Daily low-vegetation LAI (ratio) from ERA5-Land"""
    df = extract_transient_variable(stations, windows, PATHS["era5_lai_low"], band="lai_lv", lat_dim="latitude", lon_dim="longitude", preprocess=_era5_fix_coords)
    df["lai_low_vegetation_ratio"] = df.pop("lai_lv")
    return df


def step_era5_lai_high(stations, windows):
    """Daily high-vegetation LAI (ratio) from ERA5-Land"""
    df = extract_transient_variable(stations, windows, PATHS["era5_lai_high"], band="lai_hv", lat_dim="latitude", lon_dim="longitude", preprocess=_era5_fix_coords)
    df["lai_high_vegetation_ratio"] = df.pop("lai_hv")
    return df


def step_gpcc(stations, windows):
    """Precipitation (mm/yr-eq) from GPCC (natively month resolution)"""
    df = extract_transient_variable(stations, windows, PATHS["gpcc"], band="precip", mask_period="M")
    df["precipitation_gpcc_mm_year-1"] = df.pop("precip") * 12
    return df


def step_abstraction_industrial(stations, windows):
    """Industrial water withdrawal (m3/yr) from ISIMIP (natively annual resolution"""
    df = extract_transient_variable(stations, windows, PATHS["abstract_ind"], band="indww", mask_period="Y")
    df = df.rename(columns={"indww": "withdrawal_industrial_m3_year-1"})
    return df


def step_abstraction_domestic(stations, windows):
    """Domestic water withdrawal (m3/yr) from ISIMIP (natively annual resolution"""
    df = extract_transient_variable(stations, windows, PATHS["abstract_dom"], band="domww", mask_period="Y")
    df = df.rename(columns={"domww": "withdrawal_domestic_m3_year-1"})
    return df


def step_landuse_totals(stations, windows):
    """Cropland, pasture and forest land-use fractions (0-1), natively annualy resolution"""
    rename_map = {
        "cropland_irrigated": "cropland_irrigated_fraction",
        "cropland_rainfed": "cropland_rainfed_fraction",
        "pastures": "pastures_fraction",
        "forests_and_natural_vegetation": "forests_natural_vegetation_fraction",
    }
    out = None
    for band in rename_map:
        df = extract_transient_variable(stations, windows, PATHS["lu_totals"], band=band, mask_period="Y")
        out = df if out is None else out.merge(df, on=[STATION_ID_COL, "date"], how="outer")
    return out.rename(columns=rename_map)


def step_landuse_urban(stations, windows):
    """Urban area fraction (0-1), natively annual resolution."""
    df = extract_transient_variable(stations, windows, PATHS["lu_urban"], band="urbanareas", mask_period="Y")
    df = df.rename(columns={"urbanareas": "urban_area_fraction"})
    return df


def step_snow_cover_days(stations, windows):
    """Days per year with measurable snow cover, requires era5_snow_depth"""
    snow_path = output_path("atmosphere", "era5_snow_depth")
    if not snow_path.exists():
        raise RuntimeError(f"requires era5_snow_depth - check for output at {snow_path}")
    snow = pd.read_parquet(snow_path)
    snow = snow[snow[STATION_ID_COL].isin(stations[STATION_ID_COL])].copy()
    snow["year"] = pd.to_datetime(snow["date"]).dt.year

    def _year_count(group):
        if group["snow_depth_m"].notna().all():
            return int((group["snow_depth_m"] != 0).sum())
        return np.nan

    yearly = (
        snow.groupby([STATION_ID_COL, "year"])
        .apply(_year_count, include_groups=False)
        .reset_index(name="days_with_snow_cover_days_year-1")
    )
    yearly["date"] = pd.to_datetime(yearly.pop("year"), format="%Y")
    return yearly


###---------------------------------------------------###
###                 Output functions                  ###
###---------------------------------------------------###

def output_path(category, name):
    return Path(OUTPUT_ROOT) / category / f"{name}.parquet"


def load_existing_station_ids(path):
    """check which gwms_id's have already been processed (if any)"""
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=[STATION_ID_COL])[STATION_ID_COL])


def stations_needing_timeseries(stations, windows, path, mask_period=None):
    """check which gwms_id's need (re)processing (i.e. new stations, or those with observations that now exceed the pre-existing time window"""
    if not path.exists():
        return stations, windows

    existing = pd.read_parquet(path, columns=[STATION_ID_COL, "date"])
    coverage = existing.groupby(STATION_ID_COL)["date"].agg(existing_start="min", existing_end="max")
    merged = windows.merge(coverage, on=STATION_ID_COL, how="left")

    need_start, need_end = merged["window_start"], merged["window_end"]
    have_start, have_end = merged["existing_start"], merged["existing_end"]
    if mask_period is not None:
        need_start = need_start.dt.to_period(mask_period).dt.start_time
        need_end = need_end.dt.to_period(mask_period).dt.end_time
        have_start = have_start.dt.to_period(mask_period).dt.start_time
        have_end = have_end.dt.to_period(mask_period).dt.end_time

    stale = merged["existing_start"].isna() | (need_start < have_start) | (need_end > have_end)
    stale_ids = set(merged.loc[stale, STATION_ID_COL])
    return stations[stations[STATION_ID_COL].isin(stale_ids)], windows[windows[STATION_ID_COL].isin(stale_ids)]


def merge_with_existing(new_df, path, dedup_on):
    """Append new rows onto previous output"""
    if not path.exists():
        return new_df
    existing = pd.read_parquet(path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    return combined.drop_duplicates(subset=dedup_on, keep="last")


def export_table(df, path):
    """write output to .csv and .parquet"""
    if df is None or df.empty:
        print(f"  no new stations processed for {path.stem}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    df.to_parquet(path, index=False)
    print(f"  Finished {path.stem}, earth system variables written to {path.parent}/{path.stem}.csv and {path.parent}/{path.stem}.parquet")


def run_step(name, category, func, is_timeseries, mask_period=None):
    """Generic runner for processing earth system variables"""
    print(f"{name}")
    path = output_path(category, name)
    stations = load_stations()

    if is_timeseries:
        windows = compute_station_windows(stations)
        todo, todo_windows = stations, windows
        if SKIP_EXISTING_STATIONS:
            todo, todo_windows = stations_needing_timeseries(stations, windows, path, mask_period=mask_period)
        if not len(todo):
            return
        df = func(todo, todo_windows)
        df = merge_with_existing(df, path, dedup_on=[STATION_ID_COL, "date"])
    else:
        todo = stations
        if SKIP_EXISTING_STATIONS:
            done = load_existing_station_ids(path)
            todo = stations[~stations[STATION_ID_COL].isin(done)]
        if not len(todo):
            return
        df = func(todo)
        df = merge_with_existing(df, path, dedup_on=[STATION_ID_COL])

    export_table(df, path)


###---------------------------------------------------###
###                    Runner                         ###
###---------------------------------------------------###
#map variable name to: category, function, is_timeseries, mask_period)
STEPS = {
    #static (category, function, is_timeseries)
    "koeppen_geiger": ("atmosphere", step_koeppen_geiger, False),
    "hydrobelt": ("atmosphere", step_hydrobelt, False),
    "dem": ("geosphere", step_dem, False),
    "slope": ("geosphere", step_slope, False),
    "glim": ("geosphere", step_glim, False),
    "permeability": ("geosphere", step_permeability, False),
    "porosity": ("geosphere", step_porosity, False),
    "soil_texture": ("geosphere", step_soil_texture, False),
    "soil_ksat": ("geosphere", step_soil_ksat, False),
    "dist_streams": ("hydrosphere", step_dist_streams, False),
    "drainage_density": ("hydrosphere", step_drainage_density, False),
    "glacier_permafrost": ("hydroatlas", step_glacier_permafrost, False),
    "ggde": ("biosphere", step_ggde, False),
    "groundwaterscapes": ("anthroposphere", step_groundwaterscapes, False),
    #time series (category, function, is_timeseries, mask_period)
    "mswep": ("atmosphere", step_mswep, True, None),
    "ndvi": ("biosphere", step_ndvi, True, None),
    "gleam_potential_et": ("atmosphere", step_gleam_potential_et, True, None),
    "gleam_actual_et": ("atmosphere", step_gleam_actual_et, True, None),
    "gleam_interception": ("atmosphere", step_gleam_interception, True, None),
    "era5_pet": ("atmosphere", step_era5_pet, True, None),
    "era5_temperature": ("atmosphere", step_era5_temperature, True, None),
    "era5_snow_depth": ("atmosphere", step_era5_snow_depth, True, None),
    "era5_lai_low": ("atmosphere", step_era5_lai_low, True, None),
    "era5_lai_high": ("atmosphere", step_era5_lai_high, True, None),
    "gpcc": ("atmosphere", step_gpcc, True, "M"),
    "abstraction_industrial": ("anthroposphere", step_abstraction_industrial, True, "Y"),
    "abstraction_domestic": ("anthroposphere", step_abstraction_domestic, True, "Y"),
    "landuse_totals": ("anthroposphere", step_landuse_totals, True, "Y"),
    "landuse_urban": ("anthroposphere", step_landuse_urban, True, "Y"),
    "snow_cover_days": ("atmosphere", step_snow_cover_days, True, "Y"),  #dependency on era5_snow_depth
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 derive_GROW-Qual_esv.py <step> [<step> ...]   (or --all)")
        sys.exit(1)

    names = list(STEPS) if sys.argv[1] == "--all" else sys.argv[1:]

    for name in names:
        if name not in STEPS:
            print(f"Unknown step: {name}")
            sys.exit(1)

    for name in names:
        run_step(name, *STEPS[name])


if __name__ == "__main__":
    main()
