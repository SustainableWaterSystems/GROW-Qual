# -*- coding: utf-8 -*-
"""
Integrate the GROW earth system variables into GROW-Qual/zarr
- python3 integrate_esv_into_zarr.py <step>
- python3 integrate_esv_into_zarr.py --all
Dependencies: pandas, numpy, xarray, zarr, pyarrow.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                          ###
###---------------------------------------------------###

ZARR_PATH = "/gpfs/work4/0/dynql/GROW-Qual/GROW-Qual.zarr"
OUTPUT_ESV_ROOT = "/gpfs/work4/0/dynql/GROW-Qual/auxiliary/output_esv/"
STATION_ID_COL = "gwms_id"

ZARR_CHUNKS = {
    "time": 16710,
    "gwms_id": 2500,
} #aligned with GROW-Qual.zarrs own chunking


###---------------------------------------------------###
###                Variable registry                   ###
###---------------------------------------------------###

STATIC_VARS = {
    "koeppen_geiger": "atmosphere",
    "hydrobelt": "atmosphere",
    "dem": "geosphere",
    "slope": "geosphere",
    "glim": "geosphere",
    "permeability": "geosphere",
    "porosity": "geosphere",
    "soil_texture": "geosphere",
    "soil_ksat": "geosphere",
    "dist_streams": "hydrosphere",
    "drainage_density": "hydrosphere",
    "glacier_permafrost": "hydroatlas",
    "ggde": "biosphere",
    "groundwaterscapes": "anthroposphere",
}

TIMESERIES_VARS = {
    "mswep": ("atmosphere", "D"),
    "ndvi": ("biosphere", "D"),
    "gleam_potential_et": ("atmosphere", "D"),
    "gleam_actual_et": ("atmosphere", "D"),
    "gleam_interception": ("atmosphere", "D"),
    "era5_pet": ("atmosphere", "D"),
    "era5_temperature": ("atmosphere", "D"),
    "era5_snow_depth": ("atmosphere", "D"),
    "era5_lai_low": ("atmosphere", "D"),
    "era5_lai_high": ("atmosphere", "D"),
    "gpcc": ("atmosphere", "M"),
    "abstraction_industrial": ("anthroposphere", "Y"),
    "abstraction_domestic": ("anthroposphere", "Y"),
    "landuse_totals": ("anthroposphere", "Y"),
    "landuse_urban": ("anthroposphere", "Y"),
    "snow_cover_days": ("atmosphere", "Y"),
}


###---------------------------------------------------###
###           Zarr coordinate helpers                 ###
###---------------------------------------------------###

def load_zarr_coords():
    """Load gwms_id order and time axis in GROW-Qual.zarr"""
    ds = xr.open_zarr(ZARR_PATH, consolidated=False)
    gwms_ids = ds["gwms_id"].values.astype(str)
    dates = pd.DatetimeIndex(ds["time"].values)
    ds.close()
    return gwms_ids, dates


def write_variable(data_array, name, dims, chunks):
    """Add a variable GROW-Qual.zarr, overwriting existing data (used for small, gwms_id-only variables)"""
    group = zarr.open_group(ZARR_PATH, mode="r+")
    if name in group:
        del group[name]
    ds = xr.Dataset({name: (dims, data_array)})
    ds.to_zarr(ZARR_PATH, mode="a", encoding={name: {"chunks": chunks}}, consolidated=False)


def create_timeseries_variable(name, shape, chunks, dtype="float32"):
    """Pre-create a (gwms_id, time)-shaped zarr array to be filled in station-chunks."""
    group = zarr.open_group(ZARR_PATH, mode="r+")
    if name in group:
        del group[name]
    arr = group.create_array(name, shape=shape, chunks=chunks, dtype=dtype, fill_value=np.nan)
    arr.attrs["_ARRAY_DIMENSIONS"] = ["gwms_id", "time"]
    return arr


###---------------------------------------------------###
###         Convert from native resolution to daily   ###
###---------------------------------------------------###

def broadcast_to_daily(df, value_col, gwms_ids, daily_dates, native_freq):
    """Daily values direct map, but for natively monthly and annual variables broadcast this to a daily axis"""
    wide = df.pivot(index=STATION_ID_COL, columns="date", values=value_col)
    wide = wide.reindex(index=gwms_ids)

    if native_freq == "D":
        wide = wide.reindex(columns=daily_dates)
        return wide.values.astype("float32")

    period_starts = daily_dates.to_period(native_freq).start_time
    col_position = {c: i for i, c in enumerate(wide.columns)}
    day_to_col = np.array([col_position.get(p, -1) for p in period_starts])

    values = wide.values.astype("float32")
    out = np.full((values.shape[0], len(daily_dates)), np.nan, dtype="float32")
    valid = day_to_col >= 0
    out[:, valid] = values[:, day_to_col[valid]]
    return out


###---------------------------------------------------###
###           Per-variable integration                ###
###---------------------------------------------------###

def integrate_static_variable(name, category):
    """Read parquet for static variable and add to .zarr (on gwms_id) - unchanged, small enough to hold in memory"""
    path = Path(OUTPUT_ESV_ROOT) / category / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"output not found at {path}, run derive_GROW-Qual_esv.py {name}")

    df = pd.read_parquet(path)
    gwms_ids, _ = load_zarr_coords()
    df = df.set_index(STATION_ID_COL).reindex(gwms_ids)

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype="float32")
        else:
            values = series.fillna("").astype(str).to_numpy(dtype="U100")
        write_variable(values, col, ["gwms_id"], (ZARR_CHUNKS["gwms_id"],))
        print(f"  wrote {col} (dims: gwms_id)")


def integrate_timeseries_variable(name, category, native_freq):
    """Read a time-series parquet and add to .zarr (on gwms_id and time)."""
    path = Path(OUTPUT_ESV_ROOT) / category / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"output not found at {path}, run derive_GROW-Qual_esv.py {name}")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    gwms_ids, daily_dates = load_zarr_coords()

    value_cols = [c for c in df.columns if c not in (STATION_ID_COL, "date")]
    for col in value_cols:
        df[col] = df[col].astype("float32") 

    rank = pd.Categorical(df[STATION_ID_COL], categories=gwms_ids).codes
    known = rank >= 0
    df = df.loc[known].assign(_rank=rank[known]).sort_values("_rank")
    rank_values = df["_rank"].to_numpy()

    chunk_size = ZARR_CHUNKS["gwms_id"]
    n_ids = len(gwms_ids)
    arrs = {
        col: create_timeseries_variable(col, (n_ids, len(daily_dates)), (chunk_size, ZARR_CHUNKS["time"]))
        for col in value_cols
    }

    for start in range(0, n_ids, chunk_size):
        end = min(start + chunk_size, n_ids)
        lo, hi = np.searchsorted(rank_values, [start, end])
        block_df = df.iloc[lo:hi]
        block_ids = gwms_ids[start:end]
        for col in value_cols:
            arrs[col][start:end, :] = broadcast_to_daily(
                block_df[[STATION_ID_COL, "date", col]], col, block_ids, daily_dates, native_freq
            )

    for col in value_cols:
        print(f"  wrote {col} (dims: gwms_id, time)")


###---------------------------------------------------###
###                  Runner & CLI                      ###
###---------------------------------------------------###

def run_step(name):
    print(f"{name}")
    if name in STATIC_VARS:
        integrate_static_variable(name, STATIC_VARS[name])
    else:
        category, native_freq = TIMESERIES_VARS[name]
        integrate_timeseries_variable(name, category, native_freq)


def main():
    all_names = list(STATIC_VARS) + list(TIMESERIES_VARS)
    args = sys.argv[1:]

    if not args:
        print("Usage: python3 integrate_esv_into_zarr.py <step> (or --all)")
        sys.exit(1)

    is_full_run = args[0] == "--all"
    names = all_names if is_full_run else args

    for name in names:
        if name not in all_names:
            print(f"Unknown step: {name}")
            sys.exit(1)

    failures = []
    for name in names:
        try:
            run_step(name)
        except Exception as exc:
            print(f"  {name} failed: {exc}\n")
            failures.append(name)

    if failures:
        print(f"Done, but {len(failures)} step(s) failed: {', '.join(failures)}")
        print("Not consolidating zarr metadata. First need to fix any failures listed above and re-run those steps")
        sys.exit(1)

    print("All steps integrated successfully.")
    if is_full_run:
        print("Consolidating zarr metadata...")
        zarr.consolidate_metadata(ZARR_PATH)
        print("Finished adding earth system variables to GROW-Qual.zarr!")
    else:
        print("This was a partial run, remember to consolidate zarr metadata once all variables are integrated")


if __name__ == "__main__":
    main()
