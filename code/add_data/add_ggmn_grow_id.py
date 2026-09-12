# -*- coding: utf-8 -*-
import os
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

#directories and files
input_dir_gwms = "/gpfs/work4/0/dynql/GROW-Qual/"
gwms_site_info_file = os.path.join(input_dir_gwms, "gwms_site_info.csv") #output from add_gwq_dataset.py
grow_site_info_file = os.path.join(input_dir_gwms, "auxiliary", "GROW", "grow_site_info.csv") #output from process_grow_zarr.py
ggmn_site_info_file = os.path.join(input_dir_gwms, "auxiliary", "GGMN", "ggmn_site_info.csv") #output from process_ggmn_gwl.py

#constraining variable for matching gwms_id to ggmn_id and grow_id
CONSTRAINT_LAYER = "/gpfs/work4/0/dynql/GROW-Qual/auxiliary/input_esv/geosphere/GLiM/GLiM.gdb"
CONSTRAINT_LAYER_NAME = "GLiM_export"
CONSTRAINT_ID_COL = "Litho"

FID_COL = "fid_poly"
MAX_GROW_DIST_KM = 1.0


###---------------------------------------------------###
###             Match gwms_id with grow_id            ###
###---------------------------------------------------###

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_inputs():
    """Load gwms,ggmn and grow site_info, in addition to a constraint polygon layer."""
    print("Loading gwms_site info.csv...")
    gwms_sites = pd.read_csv(gwms_site_info_file,dtype={"gwms_id": str, "gwms_lat": float, "gwms_lon": float})
    
    print("Loading ggmn_site_info.csv...")
    ggmn_sites = pd.read_csv(ggmn_site_info_file,dtype={"ggmn_id": str, "ggmn_lat": float, "ggmn_lon": float},usecols=["ggmn_id", "ggmn_lat", "ggmn_lon", "has_gwl"])
    ggmn_sites = ggmn_sites.dropna(subset=["ggmn_lat", "ggmn_lon"])
    ggmn_sites = ggmn_sites[ggmn_sites["has_gwl"] == "yes"].drop(columns=["has_gwl"])

    print("Loading grow_site_info.csv...")
    grow_sites = pd.read_csv(grow_site_info_file,usecols=["grow_id", "ggmn_id", "grow_lat", "grow_lon"],dtype={"ggmn_id": str})
    grow_sites = grow_sites.dropna(subset=["grow_lat", "grow_lon"])
    
    print(f"Loading constraint layer for spatial matching...")
    constraint = (gpd.read_file(CONSTRAINT_LAYER, layer=CONSTRAINT_LAYER_NAME) if CONSTRAINT_LAYER_NAME
                 else gpd.read_file(CONSTRAINT_LAYER))
    constraint["geometry"] = constraint["geometry"].buffer(0)
    constraint = constraint.copy()
    constraint[FID_COL] = constraint.index
    
    return gwms_sites, ggmn_sites, grow_sites, constraint


def assign_constraint_polygon(sites_gdf, constraint, id_col, label):
    """Spatial-join each site to its containing constraint polygon"""
    constraint_proj = constraint.to_crs(epsg=3857)
    sites_proj = sites_gdf.to_crs(epsg=3857)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joined = gpd.sjoin(
            sites_proj[[id_col, "geometry"]],
            constraint_proj[[CONSTRAINT_ID_COL, FID_COL, "geometry"]],
            how="left", predicate="intersects"
        ).groupby(id_col).first().reset_index()

    n_in = joined[FID_COL].notna().sum()
    return joined[[id_col, CONSTRAINT_ID_COL, FID_COL]]


def tier1_same_coord(gwms_sites, target_sites, target_id_col, target_lat_col,target_lon_col, label, coord_decimals=5):
    """Exact match on rounded (lat, lon) between gwms and target sites."""
    print("Matching sites based on same (rounded) coordinates...")
    target = target_sites.copy()
    target["coord_id"] = (target[target_lat_col].round(coord_decimals).astype(str) + "_" + target[target_lon_col].round(coord_decimals).astype(str))

    coord_counts = target["coord_id"].value_counts()
    ambiguous_coords = coord_counts[coord_counts > 1]
    ambiguous_ids = set(target.loc[target["coord_id"].isin(ambiguous_coords.index), target_id_col])
    coord_to_id = target.drop_duplicates("coord_id").set_index("coord_id")[target_id_col].to_dict()

    gwms = gwms_sites.copy()
    gwms["coord_id"] = (gwms["gwms_lat"].round(coord_decimals).astype(str) + "_" + gwms["gwms_lon"].round(coord_decimals).astype(str))
    gwms[target_id_col] = gwms["coord_id"].map(coord_to_id)

    mask = gwms[target_id_col].notna()
    t1 = gwms.loc[mask, ["gwms_id", target_id_col]].copy()
    t1["match_tier"] = 1
    t1["distance_m"] = 0.0
    return t1, ~mask, ambiguous_ids


def tier2_nearest_in_polygon(gwms_sites, unmatched_mask, target_sites, target_id_col, target_lat_col, target_lon_col, gwq_constraint, target_constraint, label):
    """Nearest target site within the same constraint polygon"""
    print("Matching proximate sites with the same constraining polygons...")
    
    gwq_unmatched = gwms_sites[unmatched_mask].merge(
        gwq_constraint[["gwms_id", FID_COL]], on="gwms_id", how="left")
    target_fid = target_sites.merge(
        target_constraint[[target_id_col, FID_COL]], on=target_id_col, how="left")

    gwq_in_polygon = gwq_unmatched[gwq_unmatched[FID_COL].notna()].copy()
    
    gwq_gdf = gpd.GeoDataFrame(
        gwq_in_polygon,
        geometry=gpd.points_from_xy(gwq_in_polygon["gwms_lon"], gwq_in_polygon["gwms_lat"]),
        crs="EPSG:4326").to_crs(epsg=3857)
    target_gdf = gpd.GeoDataFrame(
        target_fid[target_fid[FID_COL].notna()],
        geometry=gpd.points_from_xy(target_fid[target_fid[FID_COL].notna()][target_lon_col],
                                     target_fid[target_fid[FID_COL].notna()][target_lat_col]),
        crs="EPSG:4326").to_crs(epsg=3857)

    results = []
    for fid in tqdm(gwq_gdf[FID_COL].unique(), desc=f"  Polygons ({label})"):
        gwq_in = gwq_gdf[gwq_gdf[FID_COL] == fid]
        tgt_in = target_gdf[target_gdf[FID_COL] == fid]
        if tgt_in.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            joined = gpd.sjoin_nearest(
                gwq_in[["gwms_id", "geometry"]], tgt_in[[target_id_col, "geometry"]],
                how="left", distance_col="distance_m")
        joined = joined.sort_values("distance_m").groupby("gwms_id").first().reset_index()
        results.append(joined[["gwms_id", target_id_col, "distance_m"]])

    if not results:
        return pd.DataFrame()

    t2 = pd.concat(results, ignore_index=True)
    t2["match_tier"] = 2
    return t2


def match_two_tier(gwms_sites, constraint, gwq_constraint, target_sites, target_id_col, target_lat_col, target_lon_col, label):
    """Tier 1 matching followed by Tier 2 matching"""
    t1, unmatched, ambiguous_ids = tier1_same_coord(gwms_sites, target_sites, target_id_col, target_lat_col, target_lon_col, label)

    target_constraint = assign_constraint_polygon(
        gpd.GeoDataFrame(target_sites[[target_id_col]],
                         geometry=gpd.points_from_xy(target_sites[target_lon_col],
                                                      target_sites[target_lat_col]),
                         crs="EPSG:4326"),
        constraint, id_col=target_id_col, label=f"{label} stations")

    t2 = tier2_nearest_in_polygon(gwms_sites, unmatched, target_sites, target_id_col, target_lat_col, target_lon_col, gwq_constraint, target_constraint, label)

    parts = [t1]
    if not t2.empty:
        parts.append(t2)
    matches = pd.concat(parts, ignore_index=True)
    print(f"  Total: {len(matches):,} / {len(gwms_sites):,} gwms sites matched")
    return matches, ambiguous_ids

def _finalize_matches(matches, distance_col):
    """Rename match_two_tier's generic distance_m to a target-specific name and drop the internal tier flag."""
    return matches.rename(columns={"distance_m": distance_col}).drop(columns=["match_tier"])


###---------------------------------------------------###
###           Match grow_id with ggmn_id              ###
###---------------------------------------------------###

def link_ggmn_id_from_grow(site_info, ggmn_sites, grow_sites, ambiguous_grow_ids):
    """Merge grow_id to ggmn_id_from_grow (via GROW's own ggmn_id crosswalk column); ambiguous grow_id matches get an additional distance check."""
    print("Linking grow_id to ggmn_id_from_grow...")
    ggmn = grow_sites[["grow_id", "ggmn_id", "grow_lat", "grow_lon"]].rename(columns={"ggmn_id": "ggmn_id_from_grow"})
    ggmn_valid = ggmn.dropna(subset=["ggmn_id_from_grow"])
    merged = site_info.merge(ggmn_valid, on="grow_id", how="left")

    has_ggmn = merged["ggmn_id_from_grow"].notna()
    needs_check = has_ggmn & merged["grow_id"].isin(ambiguous_grow_ids)

    if needs_check.any():
        ggmn_coords = ggmn_sites.set_index("ggmn_id")[["ggmn_lat", "ggmn_lon"]]
        check = merged.loc[needs_check].join(ggmn_coords, on="ggmn_id_from_grow")
        dist = haversine_km(check["ggmn_lat"], check["ggmn_lon"],
                            check["grow_lat"], check["grow_lon"])
        bad_idx = check.index[dist > MAX_GROW_DIST_KM]
        merged.loc[bad_idx, "ggmn_id_from_grow"] = np.nan

    merged = merged.drop(columns=["grow_lat", "grow_lon"])
    n_matched = merged["ggmn_id_from_grow"].notna().sum()
    print(f"  {n_matched:,} / {len(merged):,} gwms sites linked to a ggmn_id_from_grow")
    return merged


###---------------------------------------------------###
###                     Main                          ###
###---------------------------------------------------###

def main():
    print("Linking gwms_id, grow_id and ggmn_id...")

    gwms_sites, ggmn_sites, grow_sites, constraint = load_inputs()
    gwq_constraint = assign_constraint_polygon(
        gpd.GeoDataFrame(gwms_sites[["gwms_id"]],
                         geometry=gpd.points_from_xy(gwms_sites["gwms_lon"], gwms_sites["gwms_lat"]),
                         crs="EPSG:4326"),
        constraint, id_col="gwms_id", label="gwms sites")

    #match gwms_id with grow_id
    grow_matches_direct, ambiguous_grow_ids = match_two_tier(gwms_sites, constraint, gwq_constraint, grow_sites, "grow_id", "grow_lat", "grow_lon", "GROW (direct)")
    grow_matches_direct = _finalize_matches(grow_matches_direct, "grow_distance_m")

    #match gwms_id with ggmn_id, both derived directly from grow_id and also independently
    ggmn_matches, _ = match_two_tier(gwms_sites, constraint, gwq_constraint, ggmn_sites, "ggmn_id", "ggmn_lat", "ggmn_lon", "ggmn")
    ggmn_matches = _finalize_matches(ggmn_matches, "ggmn_distance_m")

    print("Updating gwms_site_info.csv...")
    site_info = pd.read_csv(gwms_site_info_file, dtype={"gwms_id": str})

    drop_cols = ["ggmn_station_id", "gwl_station_id", "grow_id", "grow_distance_m", "ggmn_id_from_grow", "ggmn_id", "ggmn_distance_m"]
    site_info = site_info.drop(columns=[c for c in drop_cols if c in site_info.columns])

    site_info = site_info.merge(grow_matches_direct, on="gwms_id", how="left")
    site_info = link_ggmn_id_from_grow(site_info, ggmn_sites, grow_sites, ambiguous_grow_ids)
    site_info = site_info.merge(ggmn_matches, on="gwms_id", how="left")

    site_info.to_csv(gwms_site_info_file, index=False)

    print("Finished adding grow_id and ggmn_id to gwms_site_info.csv!")


if __name__ == "__main__":
    main()
