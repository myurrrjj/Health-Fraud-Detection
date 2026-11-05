from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from fraud_detection.config import FEATURE_COLUMNS, FEATURE_LABELS, MODEL_PATH
from fraud_detection.modeling import load_model_bundle, train_and_save_model


st.set_page_config(page_title="Medicare Provider Risk Explorer", layout="centered")

st.title("Medicare Provider Risk Explorer")
st.write(
    "Estimate how closely provider-level summary statistics match the "
    "PotentialFraud labels in the training dataset."
)
st.info(
    "Educational prototype only. The output is a model score for a dataset label, "
    "not a determination that a provider committed fraud."
)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    if MODEL_PATH.is_file():
        return load_model_bundle()
    return train_and_save_model()


try:
    with st.spinner("Loading or training the provider model..."):
        bundle = get_model_bundle()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
except (ValueError, RuntimeError) as error:
    st.error(f"Could not prepare the model: {error}")
    st.stop()

model = bundle["model"]
metrics = bundle["metrics"]

with st.sidebar:
    st.subheader("Held-out provider evaluation")
    st.caption("These metrics are from a provider-level split of the supplied dataset.")
    st.metric("Providers used", f"{metrics['provider_count']:,}")
    st.metric("Held-out providers", f"{metrics['test_provider_count']:,}")
    for key in ("precision", "recall", "f1", "roc_auc"):
        value = metrics.get(key)
        st.metric(key.replace("_", " ").title(), "N/A" if value is None else f"{value:.3f}")

st.subheader("Provider summary")
st.caption("Enter aggregate values for one provider using the source dataset's units.")

with st.form("provider_summary"):
    left, right = st.columns(2)
    with left:
        claim_count = st.number_input("Total claims", min_value=0, value=100, step=1)
        average_amount = st.number_input(
            "Average claim reimbursement", min_value=0.0, value=5000.0, step=100.0
        )
        total_amount = st.number_input(
            "Total claim reimbursement", min_value=0.0, value=500000.0, step=1000.0
        )
        unique_beneficiaries = st.number_input(
            "Unique beneficiaries", min_value=0, value=80, step=1
        )
        inpatient_share = st.number_input(
            "Inpatient claim share (0 to 1)",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.05,
        )
    with right:
        average_duration = st.number_input(
            "Average claim duration (days)", min_value=0.0, value=1.0, step=1.0
        )
        average_age = st.number_input(
            "Average beneficiary age", min_value=0.0, max_value=120.0, value=65.0, step=1.0
        )
        average_chronic = st.number_input(
            "Average chronic-condition count", min_value=0.0, value=2.0, step=0.5
        )
        average_deductible = st.number_input(
            "Average deductible paid", min_value=0.0, value=500.0, step=50.0
        )
    submitted = st.form_submit_button("Estimate provider risk", type="primary")

if submitted:
    input_values = {
        "provider_claim_count": claim_count,
        "provider_avg_claim_amount": average_amount,
        "provider_total_claim_amount": total_amount,
        "provider_unique_beneficiaries": unique_beneficiaries,
        "provider_inpatient_share": inpatient_share,
        "provider_avg_claim_duration_days": average_duration,
        "provider_avg_beneficiary_age": average_age,
        "provider_avg_chronic_conditions": average_chronic,
        "provider_avg_deductible_amount": average_deductible,
    }
    input_frame = pd.DataFrame(
        [[input_values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
        dtype="float64",
    )
    positive_class_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(input_frame)[0, positive_class_index])
    threshold = float(bundle.get("threshold", 0.5))

    st.divider()
    st.metric("Estimated probability of the PotentialFraud label", f"{probability:.1%}")
    st.progress(float(np.clip(probability, 0.0, 1.0)))
    if probability >= threshold:
        st.warning(
            "This provider summary is above the model's demonstration threshold. "
            "Review the evidence; this score is not a fraud finding."
        )
    else:
        st.success("This provider summary is below the model's demonstration threshold.")

    st.subheader("Feature contributions")
    st.caption(
        "Positive values push the model score toward PotentialFraud=Yes; "
        "negative values push it away. These are model contributions, not causal effects."
    )
    try:
        contributions = np.asarray(
            model.booster_.predict(input_frame, pred_contrib=True)
        )[0, :-1]
        explanation = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS[name] for name in FEATURE_COLUMNS],
                "Input value": [input_values[name] for name in FEATURE_COLUMNS],
                "Contribution": contributions,
            }
        )
        explanation = explanation.reindex(
            explanation["Contribution"].abs().sort_values(ascending=False).index
        )
        st.dataframe(explanation, hide_index=True, width="stretch")
    except (AttributeError, ValueError, TypeError) as error:
        st.caption(f"Local contribution details are unavailable: {error}")
