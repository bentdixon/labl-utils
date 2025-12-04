import polars as pl


"""
A function for combining CSV files with different column names.
Joins on 'patient_id', and optionally, 'clinical_status' if available. 
All other metrics columns follow. 
"""
def combine_csvs(csv_paths: list[str]) -> pl.DataFrame:
    """Load and join multiple CSVs on patient_id and clinical_status."""
    dataframes = [pl.read_csv(path) for path in csv_paths]

    # Sort so CSVs with clinical_status come first
    dataframes = sorted(
        dataframes,
        key=lambda df: "clinical_status" not in df.columns
    )

    # Join on patient_id, and clinical_status when available
    combined = dataframes[0]
    for df in dataframes[1:]:
        join_cols = ["patient_id"]
        if "clinical_status" in combined.columns and "clinical_status" in df.columns:
            join_cols.append("clinical_status")
        combined = combined.join(df, on=join_cols, how="inner")

    # Reorder columns: patient_id, clinical_status (if present), then metrics
    if "clinical_status" in combined.columns:
        cols = ["patient_id", "clinical_status"] + [
            c for c in combined.columns if c not in ("patient_id", "clinical_status")
        ]
        combined = combined.select(cols)

    return combined