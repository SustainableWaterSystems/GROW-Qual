# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import geopandas as gpd
import glob
from shapely.geometry import Point
from tqdm import tqdm
import warnings

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

#directories
input_dir_gwms = "/gpfs/work4/0/dynql/GROW-Qual/"
input_dir_aux_data = "/gpfs/work4/0/dynql/GROW-Qual/auxiliary/"  #shared auxiliary data
database = sys.argv[1]

#groundwater quality data file paths
site_info_file = os.path.join(input_dir_gwms, "gwms_site_info.csv")
existing_gw_data_file = os.path.join(input_dir_aux_data, "gwq_data/combined_gwms_dataset.csv")
gw_data_folder = os.path.join(input_dir_gwms, "GROW-Qual_raw", database)

# auxiliary file paths
country_identifier = pd.read_csv(
    os.path.join(input_dir_aux_data, "input_esv/anthroposphere/WorldBank/Country_Region_Econ_ID.txt"),
    sep="\t", header=0, quotechar='"'
)
country_shp = gpd.read_file(os.path.join(input_dir_aux_data, "input_esv/anthroposphere/WorldBank/WB_countries_Admin0_10m.shp"))
hydrobasin_shp = gpd.read_file(os.path.join(input_dir_aux_data, "input_esv/hydroatlas/BasinATLAS/BasinATLAS_v10_lev12.shp"))


###---------------------------------------------------###
###           Add groundwater quality data            ###
###---------------------------------------------------###

def load_site_info():
    """Load existing GROW-Qual gwms_site_info.csv, or create new gwms_site_info if needed"""

    required_columns = ["gwms_id","gwms_lat","gwms_lon","country_name","hydrobasin_level12","sources",]

    if not os.path.exists(site_info_file):
        pd.DataFrame(columns=required_columns).to_csv(site_info_file, index=False)

    df = pd.read_csv(site_info_file, dtype={
        "gwms_id": str,
        "gwms_lat": float,
        "gwms_lon": float,
        "country_name": str,
        "hydrobasin_level12": str,
        "sources": str,
    })

    return df


def load_gw_data(folder):
    """Load all raw .csvs (per variable) from the specified database folder"""

    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    df_list = []
    for f in files:
        df_list.append(pd.read_csv(f, dtype={
            "lat": float,
            "lon": float,
            "dates": str,
            "site_id": str,
            "limit_flag": str,
            "raw_obs": float,
            "raw_unit": str,
            "detection_limit": float,
            "source": str,
            "variable": str,
        }))
    df = pd.concat(df_list, ignore_index=True)
    print(f"  Loaded {len(df):,} observations from {len(files)} files")
    return df


def assign_country_hydrobasin(sites_gdf, country_shp, hydrobasin_shp):
    """Assign country name and HydroBASINS level12 via spatial join, using nearest-neighbour for sites that fall outside polygons (e.g. coastal wells, islands)."""

    country_shp    = country_shp.copy()
    hydrobasin_shp = hydrobasin_shp.copy()
    country_shp["geometry"]    = country_shp["geometry"].buffer(0)
    hydrobasin_shp["geometry"] = hydrobasin_shp["geometry"].buffer(0)

    sites_proj   = sites_gdf.to_crs(epsg=3857)
    country_proj = country_shp.to_crs(epsg=3857)
    hydro_proj   = hydrobasin_shp.to_crs(epsg=3857)

    sites_proj["country_name"]       = None
    sites_proj["hydrobasin_level12"] = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        #country (intersect)
        sites_country = gpd.sjoin(
            sites_proj, country_proj[["NAME_EN", "geometry"]],
            how="left", predicate="intersects"
        )
        sites_country = sites_country.groupby(sites_country.index).first()
        sites_proj.loc[sites_country.index, "country_name"] = sites_country["NAME_EN"]

        #hydrobasin lvl12 (intersect)
        sites_hydro = gpd.sjoin(
            sites_proj, hydro_proj[["HYBAS_ID", "geometry"]],
            how="left", predicate="intersects"
        )
        sites_hydro = sites_hydro.groupby(sites_hydro.index).first()
        sites_proj.loc[sites_hydro.index, "hydrobasin_level12"] = sites_hydro["HYBAS_ID"]

        #country (nearest neighbour fallback)
        unmatched_country = sites_proj[sites_proj["country_name"].isna()]
        if len(unmatched_country) > 0:
            nearest_country = gpd.sjoin_nearest(
                unmatched_country, country_proj[["NAME_EN", "geometry"]],
                how="left", distance_col=None)
            sites_proj.loc[nearest_country.index, "country_name"] = nearest_country["NAME_EN"]

        #hydrobasin lvl12 (nearest neighbour fallback)
        unmatched_hydro = sites_proj[sites_proj["hydrobasin_level12"].isna()]
        if len(unmatched_hydro) > 0:
            nearest_hydro = gpd.sjoin_nearest(
                unmatched_hydro, hydro_proj[["HYBAS_ID", "geometry"]],
                how="left", distance_col=None)
            sites_proj.loc[nearest_hydro.index, "hydrobasin_level12"] = nearest_hydro["HYBAS_ID"]

    return sites_proj


def merge_with_database(sites_gdf, gw_data, gw_database_site_info,country_identifier, coord_precision=5):
    """Assign gwms_ids and merge observations with database."""

    #process coordinates (round to specified precision)
    for df in [gw_database_site_info, sites_gdf]:
        df["gwms_lat"] = df["gwms_lat"].round(coord_precision)
        df["gwms_lon"] = df["gwms_lon"].round(coord_precision)
        df["coord_id"] = df["gwms_lat"].astype(str) + "_" + df["gwms_lon"].astype(str) #build coord_id, used for for deduplication

    #map existing coord_ids to gwms_ids
    existing_coord_map = (
        gw_database_site_info
        .set_index("coord_id")["gwms_id"]
        .to_dict()
    )
    sites_gdf["gwms_id"] = sites_gdf["coord_id"].map(existing_coord_map).astype("object")

    #identify new sites
    new_sites = sites_gdf[sites_gdf["gwms_id"].isna()].copy()
    new_sites = new_sites.drop_duplicates(subset="coord_id")

    if not new_sites.empty:
        #merge country FID
        new_sites = new_sites.merge(
            country_identifier[["Country", "FID"]],
            left_on="country_name", right_on="Country",
            how="left"
        )

        sites_dropped = new_sites[new_sites["FID"].isna()]
        if not sites_dropped.empty:
            new_sites = new_sites[new_sites["FID"].notna()]

        if new_sites.empty:
            return gw_database_site_info, gw_data

        new_sites["FID_str"] = new_sites["FID"].astype(int).astype(str).str.zfill(3)

        #track highest existing number per FID (as country code is used in defining gwms_id)
        existing_fid_max = (
            gw_database_site_info
            .merge(country_identifier[["Country", "FID"]], left_on="country_name",
                   right_on="Country", how="left")
            .assign(FID_str=lambda df: df["FID"].astype(int).astype(str).str.zfill(3))
            .groupby("FID_str")["gwms_id"]
            .apply(lambda x: x.str[-5:].astype(int).max() if not x.empty else 0)
            .to_dict()
        )

        #assign new gwms_ids
        assigned_ids = {}
        for _, row in new_sites.iterrows():
            fid          = row["FID_str"]
            coord_id     = row["coord_id"]
            next_num     = existing_fid_max.get(fid, 0) + 1
            existing_fid_max[fid] = next_num
            assigned_ids[coord_id] = "gwms_" + fid + str(next_num).zfill(5)

        new_sites["gwms_id"] = new_sites["coord_id"].map(assigned_ids)
        new_sites = new_sites.drop(columns=["FID", "Country", "FID_str"], errors="ignore")
        new_sites = new_sites.drop_duplicates(subset="gwms_id")

        gw_database_site_info = pd.concat([gw_database_site_info, new_sites],
                                           ignore_index=True)

    #push newly assigned gwms_ids into sites_gdf
    if not new_sites.empty:
        new_id_map = new_sites.set_index("coord_id")["gwms_id"].to_dict()
        mask = sites_gdf["gwms_id"].isna()
        sites_gdf.loc[mask, "gwms_id"] = sites_gdf.loc[mask, "coord_id"].map(new_id_map)

    #build per-site source lookup from the observations
    site_sources = (
        gw_data.groupby("site_id")["source"]
        .apply(lambda x: ";".join(sorted(x.dropna().unique())))
        .to_dict()
    )
    site_to_gwms_for_sources = sites_gdf.set_index("site_id")["gwms_id"].to_dict()

    gwms_new_sources = {}
    for site_id, srcs in site_sources.items():
        gwms_id = site_to_gwms_for_sources.get(site_id)
        if not gwms_id:
            continue
        for src in srcs.split(";"):
            gwms_new_sources.setdefault(gwms_id, set()).add(src)

    for gwms_id, new_srcs in gwms_new_sources.items():
        idx = gw_database_site_info[gw_database_site_info["gwms_id"] == gwms_id].index
        if len(idx) == 0:
            continue
        current = gw_database_site_info.loc[idx[0], "sources"]
        existing_srcs = set(str(current).split(";")) if pd.notna(current) and current != "" else set()
        merged = existing_srcs | new_srcs
        gw_database_site_info.loc[idx[0], "sources"] = ";".join(sorted(merged))

    #map site_id to gwms_id into observations
    site_to_gwms = sites_gdf.set_index("site_id")["gwms_id"].to_dict()
    gw_data["gwms_id"] = gw_data["site_id"].map(site_to_gwms).astype("object")

    obs_dropped = gw_data["gwms_id"].isna().sum()
    if obs_dropped > 0:
        gw_data = gw_data[gw_data["gwms_id"].notna()]

    #attach gwms coordinates to observations
    gw_data = gw_data.merge(
        gw_database_site_info[["gwms_id", "gwms_lat", "gwms_lon"]],
        on="gwms_id", how="left"
    )

    gw_data = gw_data[[
        "site_id", "lat", "lon", "gwms_id", "gwms_lat", "gwms_lon",
        "dates", "limit_flag", "raw_obs", "raw_unit", "detection_limit",
        "source", "variable"
    ]]

    return gw_database_site_info, gw_data


def save_results(gw_database_site_info, gw_data, database):
    """save updated site info and append observations to the combined dataset."""

    gwq_dir = os.path.join(input_dir_aux_data, "gwq_data")
    os.makedirs(gwq_dir, exist_ok=True)

    site_info_columns = ["gwms_id","gwms_lat","gwms_lon","country_name","hydrobasin_level12","sources",]

    site_info_full = gw_database_site_info[site_info_columns]
    site_info_full.to_csv(site_info_file, index=False)
    print(f"Full site info saved to: {site_info_file}")

    unique_gwms_ids = gw_data["gwms_id"].unique()
    filtered_site_info = site_info_full[site_info_full["gwms_id"].isin(unique_gwms_ids)]
    filtered_file = os.path.join(gwq_dir, f"gwms_site_info_for_{database}.csv")
    filtered_site_info.to_csv(filtered_file, index=False)
    print(f"Filtered site info saved to: {filtered_file}")

    if os.path.exists(existing_gw_data_file):
        existing_cols = pd.read_csv(existing_gw_data_file, nrows=0).columns.tolist()
        gw_data = gw_data[existing_cols]
        gw_data.to_csv(existing_gw_data_file, mode="a", header=False, index=False)
    else:
        gw_data.to_csv(existing_gw_data_file, index=False)
        

###---------------------------------------------------###
###                     Main                          ###
###---------------------------------------------------###

def main():
    print("Loading existing site info...")
    gw_database_site_info = load_site_info()

    print(f"Loading new groundwater quality data from: {gw_data_folder}...")
    gw_data = load_gw_data(gw_data_folder)

    unique_sites = gw_data.drop_duplicates("site_id").copy()
    unique_sites["gwms_lat"] = unique_sites["lat"]
    unique_sites["gwms_lon"] = unique_sites["lon"]
    geometry  = [Point(xy) for xy in zip(unique_sites["lon"], unique_sites["lat"])]
    sites_gdf = gpd.GeoDataFrame(unique_sites, geometry=geometry, crs="EPSG:4326")

    print("Assigning country and hydrobasin info...")
    sites_gdf = assign_country_hydrobasin(sites_gdf, country_shp, hydrobasin_shp)
    print(f"  Countries assigned: {sites_gdf['country_name'].notna().sum()}/{len(sites_gdf)}")
    print(f"  Hydrobasins assigned: {sites_gdf['hydrobasin_level12'].notna().sum()}/{len(sites_gdf)}")

    print("Merging site info and observations with existing database...")
    gw_database_site_info, gw_data = merge_with_database(sites_gdf, gw_data, gw_database_site_info, country_identifier)
    print(f"  Total sites in database: {len(gw_database_site_info):,}")
    print(f"  Observations after merge: {len(gw_data):,}")

    print("Saving results...")
    save_results(gw_database_site_info, gw_data, database)

    print(f"Finished processing database: {database}!")


if __name__ == "__main__":
    main()
