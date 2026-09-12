import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)

###---------------------------------------------------###
###                     Setup                         ###
###---------------------------------------------------###

input_dir = "/gpfs/work4/0/dynql/GROW-Qual/"

site_info_file = os.path.join(input_dir, "gwms_site_info.csv")
variable_list_path = os.path.join(input_dir, "gwq_variable_list.csv")
combined_gw_file = os.path.join(input_dir, "auxiliary/gwq_data/combined_gwms_dataset.csv")

#output directory for per-variable .csv files
csv_dir = os.path.join(input_dir, "gwms-csv")
os.makedirs(csv_dir, exist_ok=True)

ROS_MIN_DETECTS = 5
ROS_MIN_DETECT_FRACTION = 0.5

OUTLIER_IQR_MULTIPLIER = 5.0
OUTLIER_MIN_N = 10


###---------------------------------------------------###
###                    Functions                      ###
###---------------------------------------------------###

def round_to_sig_figs(x, sig_figs=5):
    """Round a value to a given number of significant figures."""

    if x == 0 or (isinstance(x, float) and np.isnan(x)):
        return x
    magnitude = np.floor(np.log10(np.abs(x)))
    return round(x, -int(magnitude) + (sig_figs - 1))


def detect_outliers(df_group, global_min, global_max, iqr_multiplier=3.0, min_n=10, log_transform=False):
    """Detect physical and statistical outliers."""

    mask_physical = (df_group["obs"] >= global_min) & (df_group["obs"] <= global_max)

    if len(df_group) < min_n or mask_physical.sum() < min_n:
        return mask_physical

    values = df_group.loc[mask_physical, "obs"].copy()

    if log_transform:
        values_for_iqr = np.log10(values + 1e-6)
    else:
        values_for_iqr = values

    Q1  = values_for_iqr.quantile(0.25)
    Q3  = values_for_iqr.quantile(0.75)
    IQR = Q3 - Q1

    if IQR == 0:
        return mask_physical

    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    if log_transform:
        obs_transformed = np.log10(df_group["obs"] + 1e-6)
    else:
        obs_transformed = df_group["obs"]

    mask_statistical = (obs_transformed >= lower_bound) & (obs_transformed <= upper_bound)
    return mask_physical & mask_statistical


def ros_substitution(values, is_censored, is_detected):
    """Regression on Order Statistics (ROS) imputation for left-censored observations."""

    values = np.array(values)
    is_censored = np.array(is_censored)
    is_detected = np.array(is_detected)

    detected = values[is_detected]
    censored_limits = values[is_censored]

    if len(detected) == 0:
        return values * 0.5

    log_detected = np.log(detected)

    n_detected = len(detected)
    detected_sorted_idx = np.argsort(detected)
    plotting_positions = np.arange(1, n_detected + 1) / (n_detected + 1)

    norm_quantiles = stats.norm.ppf(plotting_positions)
    slope, intercept = np.polyfit(norm_quantiles, log_detected[detected_sorted_idx], 1)

    all_values = np.concatenate([detected, censored_limits])
    sorted_idx = np.argsort(all_values)
    ranks = np.empty(len(all_values), dtype=int)
    ranks[sorted_idx] = np.arange(len(all_values))

    censored_ranks = ranks[len(detected):]
    censored_plotting_pos = (censored_ranks + 1) / (len(all_values) + 1)

    censored_norm_quantiles = stats.norm.ppf(censored_plotting_pos)
    log_censored_predicted = intercept + slope * censored_norm_quantiles
    censored_predicted = np.array([round_to_sig_figs(v, 5) for v in np.exp(log_censored_predicted)])

    imputed_values = values.copy()
    imputed_values[is_censored] = censored_predicted
    return imputed_values


def apply_ros_or_substitution(df_group):
    """Apply LOD imputation (ROS or direct substitution, depending on data availability)."""

    is_censored = (df_group["flag"] == "<").values
    is_outlier = (df_group["flag"] == "*").values
    is_detected = ~is_censored & ~is_outlier
    n_censored = is_censored.sum()
    n_detected = is_detected.sum()
    n_total = len(df_group)

    if n_censored == 0:
        return df_group, {"method": "none", "n_censored": 0, "n_detected": n_detected}

    df_group = df_group.copy()
    if "imputation_method" not in df_group.columns:
        df_group["imputation_method"] = ""

    working_values = df_group["obs"].copy()
    working_values[is_censored] = df_group.loc[is_censored, "detection_limit"]

    detect_fraction = n_detected / n_total
    use_ros = (n_detected >= ROS_MIN_DETECTS and detect_fraction >= ROS_MIN_DETECT_FRACTION)

    if use_ros:
        try:
            imputed = ros_substitution(working_values.values, is_censored, is_detected)
            df_group["obs"] = imputed
            df_group.loc[is_censored, "imputation_method"] = "ROS"
            method = "ros"
        except Exception as e:
            print(f"    ROS failed, using substitution: {e}")
            df_group.loc[is_censored, "obs"] = (df_group.loc[is_censored, "detection_limit"] * 0.5).apply(lambda x: round_to_sig_figs(x, 5))
            df_group.loc[is_censored, "imputation_method"] = "LOD/2"
            method = "substitution_fallback"
    else:
        df_group.loc[is_censored, "obs"] = (df_group.loc[is_censored, "detection_limit"] * 0.5).apply(lambda x: round_to_sig_figs(x, 5))
        df_group.loc[is_censored, "imputation_method"] = "LOD/2"
        method = "substitution"

    return df_group, {"method": method,"n_censored": n_censored,"n_detected": n_detected,"n_total": n_total,"censoring_fraction": n_censored / n_total,}


###-----------------------------------------------------###
###        Clean and process groundwater quality data   ###
###-----------------------------------------------------###

def clean_gw_data():
    """Clean raw groundwater quality data and save per-variable CSVs."""

    var_limits = pd.read_csv(variable_list_path)
    var_limits["variable_code"] = var_limits["variable_code"].astype(str)

    if "log_transform" not in var_limits.columns:
        var_limits["log_transform"] = False

    print("Loading combined groundwater quality dataset...")
    df_all = pd.read_csv(combined_gw_file, dtype=str)
    df_all["raw_obs"] = pd.to_numeric(df_all["raw_obs"], errors="coerce")
    df_all["detection_limit"] = pd.to_numeric(df_all["detection_limit"], errors="coerce")
    df_all["dates"] = pd.to_datetime(df_all["dates"], format="mixed", dayfirst=True, errors="coerce")
    df_all["obs"]  = df_all["raw_obs"]
    df_all["unit"] = df_all["raw_unit"]

    if "limit_flag" in df_all.columns:
        df_all = df_all.rename(columns={"limit_flag": "flag"})

    variables = df_all["variable"].dropna().unique()
    print(f"Variables to process: {sorted(variables)}\n")

    for variable_code in variables:
        df = df_all[df_all["variable"] == variable_code].copy()
        print(f"\nProcessing variable '{variable_code}' ({len(df):,} rows) ...")
        original_len = len(df)

        #step 1: keep only rows with a valid observation or a valid censored limit
        before = len(df)
        mask_valid = (
            df["obs"].notna() |
            ((df["flag"] == "<") & df["detection_limit"].notna())
        )
        df = df[mask_valid]
        print(f"  Dropped invalid obs/detection_limit rows: {before - len(df)}")

        #step 2: outlier flagging (physical bounds + per-site IQR)
        limits = var_limits[var_limits["variable_code"] == variable_code]

        if not limits.empty:
            plausible_min = limits["plausible_min"].values[0]
            plausible_max = limits["plausible_max"].values[0]

            use_log_transform = limits["log_transform"].values[0]
            if isinstance(use_log_transform, str):
                use_log_transform = use_log_transform.upper() in ["TRUE", "YES", "1", "T", "Y"]
            else:
                use_log_transform = bool(use_log_transform)

            outlier_mask  = pd.Series(False, index=df.index)
            outlier_stats = {"physical_only": 0, "statistical": 0, "physical_drops": 0, "statistical_drops": 0}

            for gwms_id, group in df.groupby("gwms_id"):
                detected_mask = group["obs"].notna()
                if detected_mask.sum() == 0:
                    continue

                detected_group  = group[detected_mask]
                group_keep_mask = detect_outliers(
                    detected_group, plausible_min, plausible_max,
                    iqr_multiplier=OUTLIER_IQR_MULTIPLIER,
                    min_n=OUTLIER_MIN_N,
                    log_transform=use_log_transform
                )

                outlier_mask.loc[detected_group.index] = ~group_keep_mask.values

                physical_drops = (
                    (detected_group["obs"] < plausible_min) |
                    (detected_group["obs"] > plausible_max)
                ).sum()
                total_drops = (~group_keep_mask).sum()

                if physical_drops > 0:
                    outlier_stats["physical_only"]  += 1
                    outlier_stats["physical_drops"] += physical_drops
                if total_drops > physical_drops:
                    outlier_stats["statistical"]       += 1
                    outlier_stats["statistical_drops"] += (total_drops - physical_drops)

            df.loc[outlier_mask & (df["flag"] != "<"), "flag"] = "*"

            #also flag censored observations with implausible detection limits
            censored_mask = (df["flag"] == "<") & df["detection_limit"].notna()
            invalid_dl = censored_mask & ((df["detection_limit"] < plausible_min) | (df["detection_limit"] > plausible_max))
            df.loc[invalid_dl, "flag"] = "*"

        #step 3: LOD imputation (ROS or LOD/2) per site
        n_censored_total = (df["flag"] == "<").sum()
        if n_censored_total > 0:
            processed_groups = []
            for gwms_id, group in df.groupby("gwms_id"):
                processed_group, _ = apply_ros_or_substitution(group)
                processed_groups.append(processed_group)
            df = pd.concat(processed_groups, ignore_index=True)

        #step 4: remove exact duplicates
        dup_cols = ["variable", "gwms_id", "dates", "unit", "obs", "flag"]
        if "imputation_method" in df.columns: dup_cols.append("imputation_method")
        df = df.drop_duplicates(subset=dup_cols)

        #step 5: remove implausible (future) dates
        today  = pd.Timestamp.today().normalize()
        df = df[df["dates"] <= today]

        #step 6: tidy columns
        if "imputation_method" not in df.columns: df["imputation_method"] = ""
        df = df[["gwms_id", "dates", "obs", "unit", "flag", "detection_limit", "imputation_method", "variable"]]

        #step 7: save output to {variable_code}.csv
        out_path = os.path.join(csv_dir, f"{variable_code}.csv")
        df.to_csv(out_path, index=False)

        print(f"  Original rows: {original_len:,}")
        print(f"  Final rows: {len(df):,}")
        print(f"  Total dropped: {original_len - len(df):,}")
        print(f"  Saved: {out_path}")


###---------------------------------------------------###
###                   Main                            ###
###---------------------------------------------------###

if __name__ == "__main__":
    print("Starting processing GROW-Qual .csvs...")
    clean_gw_data()
    print("Finished processing groundwater quality data into .csvs!")
