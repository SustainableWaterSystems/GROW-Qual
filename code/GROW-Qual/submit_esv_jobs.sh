#!/bin/bash
set -euo pipefail

PROJECT_DIR="/gpfs/work4/0/dynql/GROW-Qual"
SCRIPT="scripts/GROW-Qual/derive_GROW-Qual_esv.py"
CONDA_ENV="/gpfs/home6/ejones/.conda/envs/myenv"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

STATIC_STEPS=(
    koeppen_geiger hydrobelt dem slope glim permeability porosity
    soil_texture soil_ksat dist_streams drainage_density glacier_permafrost
    ggde groundwaterscapes
)
TIMESERIES_STEPS=(
    mswep ndvi gleam_potential_et gleam_actual_et gleam_interception
    era5_pet era5_temperature era5_snow_depth era5_lai_low era5_lai_high
    gpcc abstraction_industrial abstraction_domestic landuse_totals landuse_urban
)

submit_step() {
    local step="$1"
    shift
    sbatch --parsable \
        -t 24:00:00 -p rome -N 1 -n 128 \
        --job-name="esv_$step" \
        --output="$LOG_DIR/esv_${step}_%j.out" \
        --error="$LOG_DIR/esv_${step}_%j.err" \
        "$@" \
        --wrap="cd $PROJECT_DIR && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python3 $SCRIPT $step"
}

declare -A JOBIDS
for step in "${STATIC_STEPS[@]}" "${TIMESERIES_STEPS[@]}"; do
    jobid=$(submit_step "$step")
    echo "submitted $step as job $jobid"
    JOBIDS[$step]="$jobid"
done

#note that snow_cover_days requires output from era5_snow_depth, so there is a dependency here
jobid=$(submit_step "snow_cover_days" --dependency="afterok:${JOBIDS[era5_snow_depth]}")
echo "submitted snow_cover_days as job $jobid (after era5_snow_depth: ${JOBIDS[era5_snow_depth]})"
JOBIDS[snow_cover_days]="$jobid"

ALL_IDS=$(printf '%s:' "${JOBIDS[@]}")
echo "${ALL_IDS%:}" > "$LOG_DIR/esv_all_job_ids.txt"
echo "all job IDs written to $LOG_DIR/esv_all_job_ids.txt"
