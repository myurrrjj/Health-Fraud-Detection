from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "provider_fraud_model.joblib"
MODEL_FORMAT_VERSION = 1

TRAINING_FILES = {
    "provider_labels": "Train-1542865627584.csv",
    "beneficiaries": "Train_Beneficiarydata-1542865627584.csv",
    "inpatient_claims": "Train_Inpatientdata-1542865627584.csv",
    "outpatient_claims": "Train_Outpatientdata-1542865627584.csv",
}

FEATURE_COLUMNS = (
    "provider_claim_count",
    "provider_avg_claim_amount",
    "provider_total_claim_amount",
    "provider_unique_beneficiaries",
    "provider_inpatient_share",
    "provider_avg_claim_duration_days",
    "provider_avg_beneficiary_age",
    "provider_avg_chronic_conditions",
    "provider_avg_deductible_amount",
)

FEATURE_LABELS = {
    "provider_claim_count": "Claim count",
    "provider_avg_claim_amount": "Average claim reimbursement",
    "provider_total_claim_amount": "Total claim reimbursement",
    "provider_unique_beneficiaries": "Unique beneficiaries",
    "provider_inpatient_share": "Inpatient claim share",
    "provider_avg_claim_duration_days": "Average claim duration (days)",
    "provider_avg_beneficiary_age": "Average beneficiary age",
    "provider_avg_chronic_conditions": "Average chronic-condition count",
    "provider_avg_deductible_amount": "Average deductible paid",
}
