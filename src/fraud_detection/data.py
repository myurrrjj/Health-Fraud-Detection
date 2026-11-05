from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RAW_DATA_DIR, TRAINING_FILES


CLAIM_COLUMNS = (
    "Provider",
    "BeneID",
    "ClaimID",
    "ClaimStartDt",
    "ClaimEndDt",
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
)


def _missing_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _read_claims(path: Path, claim_type: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    _missing_columns(pd.DataFrame(columns=header), set(CLAIM_COLUMNS), path.name)
    claims = pd.read_csv(
        path,
        usecols=list(CLAIM_COLUMNS),
        dtype={"Provider": "string", "BeneID": "string", "ClaimID": "string"},
        low_memory=False,
    )
    claims["ClaimType"] = claim_type
    return claims


def load_training_tables(raw_data_dir: Path = RAW_DATA_DIR) -> dict[str, Any]:
    raw_data_dir = Path(raw_data_dir)
    paths = {key: raw_data_dir / name for key, name in TRAINING_FILES.items()}
    missing = [
        TRAINING_FILES[key]
        for key, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - data/raw/{name}" for name in missing)
        raise FileNotFoundError(
            "Training data is missing. Put these source files in data/raw/:\n" + formatted
        )

    label_path = paths["provider_labels"]
    labels = pd.read_csv(
        label_path,
        usecols=["Provider", "PotentialFraud"],
        dtype={"Provider": "string", "PotentialFraud": "string"},
        low_memory=False,
    )
    _missing_columns(labels, {"Provider", "PotentialFraud"}, label_path.name)

    beneficiary_path = paths["beneficiaries"]
    beneficiary_header = pd.read_csv(beneficiary_path, nrows=0).columns
    _missing_columns(
        pd.DataFrame(columns=beneficiary_header),
        {"BeneID", "DOB"},
        beneficiary_path.name,
    )
    chronic_columns = sorted(
        column for column in beneficiary_header if column.startswith("ChronicCond_")
    )
    beneficiary_columns = ["BeneID", "DOB", *chronic_columns]
    beneficiaries = pd.read_csv(
        beneficiary_path,
        usecols=beneficiary_columns,
        dtype={"BeneID": "string"},
        low_memory=False,
    )

    inpatient = _read_claims(paths["inpatient_claims"], "Inpatient")
    outpatient = _read_claims(paths["outpatient_claims"], "Outpatient")

    return {
        "provider_labels": labels,
        "beneficiaries": beneficiaries,
        "inpatient_claims": inpatient,
        "outpatient_claims": outpatient,
    }


def build_provider_dataset(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Build one feature row per labeled provider from the raw training tables."""
    tables = load_training_tables(raw_data_dir)
    labels = tables["provider_labels"].copy()
    beneficiaries = tables["beneficiaries"].copy()

    labels = labels.dropna(subset=["Provider"])
    if labels["Provider"].duplicated().any():
        raise ValueError("The provider-label file contains duplicate Provider IDs.")

    target = labels["PotentialFraud"].str.strip().str.casefold().map({"no": 0, "yes": 1})
    if target.isna().any():
        values = sorted(labels.loc[target.isna(), "PotentialFraud"].dropna().unique().tolist())
        raise ValueError(f"Unexpected PotentialFraud label values: {values}")
    labels["fraud"] = target.astype("int8")

    chronic_columns = [
        column for column in beneficiaries.columns if column.startswith("ChronicCond_")
    ]
    beneficiaries["DOB"] = pd.to_datetime(beneficiaries["DOB"], errors="coerce")
    for column in chronic_columns:
        beneficiaries[column] = pd.to_numeric(beneficiaries[column], errors="coerce")
    if chronic_columns:
        beneficiaries["ChronicScore"] = beneficiaries[chronic_columns].fillna(0).sum(axis=1)
    else:
        beneficiaries["ChronicScore"] = 0.0

    beneficiaries = beneficiaries.drop_duplicates(subset=["BeneID"], keep="first")
    claims = pd.concat(
        [tables["inpatient_claims"], tables["outpatient_claims"]],
        ignore_index=True,
        copy=False,
    )
    claims["ClaimStartDt"] = pd.to_datetime(claims["ClaimStartDt"], errors="coerce")
    claims["ClaimEndDt"] = pd.to_datetime(claims["ClaimEndDt"], errors="coerce")
    claims["ClaimAmount"] = pd.to_numeric(
        claims["InscClaimAmtReimbursed"], errors="coerce"
    ).fillna(0.0)
    claims["DeductibleAmount"] = pd.to_numeric(
        claims["DeductibleAmtPaid"], errors="coerce"
    ).fillna(0.0)
    claims["ClaimDurationDays"] = (
        (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days
        .clip(lower=0)
        .fillna(0)
    )

    claims = claims.merge(
        beneficiaries[["BeneID", "DOB", "ChronicScore"]],
        on="BeneID",
        how="left",
        validate="many_to_one",
    )
    age_years = (claims["ClaimStartDt"] - claims["DOB"]).dt.days / 365.2425
    claims["BeneficiaryAge"] = age_years.where(age_years.between(0, 120))
    claims["IsInpatient"] = claims["ClaimType"].eq("Inpatient").astype("int8")
    claims["_claim_count"] = 1

    aggregates = (
        claims.groupby("Provider", dropna=True, sort=False)
        .agg(
            provider_claim_count=("_claim_count", "sum"),
            provider_avg_claim_amount=("ClaimAmount", "mean"),
            provider_total_claim_amount=("ClaimAmount", "sum"),
            provider_unique_beneficiaries=("BeneID", "nunique"),
            provider_inpatient_share=("IsInpatient", "mean"),
            provider_avg_claim_duration_days=("ClaimDurationDays", "mean"),
            provider_avg_beneficiary_age=("BeneficiaryAge", "mean"),
            provider_avg_chronic_conditions=("ChronicScore", "mean"),
            provider_avg_deductible_amount=("DeductibleAmount", "mean"),
        )
        .reset_index()
    )

    dataset = labels[["Provider", "fraud"]].merge(
        aggregates,
        on="Provider",
        how="left",
        validate="one_to_one",
    )
    feature_columns = [
        "provider_claim_count",
        "provider_avg_claim_amount",
        "provider_total_claim_amount",
        "provider_unique_beneficiaries",
        "provider_inpatient_share",
        "provider_avg_claim_duration_days",
        "provider_avg_beneficiary_age",
        "provider_avg_chronic_conditions",
        "provider_avg_deductible_amount",
    ]
    dataset[feature_columns] = (
        dataset[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float64")
    )
    return dataset.sort_values("Provider", kind="stable").reset_index(drop=True)
