import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Customer Auto-Renewal Prediction Dashboard
# Uses the user's six existing trained .pkl models.
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Auto-Renewal Prediction",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "customer_churn_dataset.csv"

FEATURES = [
    "duration_days",
    "completed_deliveries",
    "delayed_deliveries",
    "delivery_ratio",
    "total_contacts",
    "complaints",
    "complaint_ratio",
    "total_web_visits",
    "orders",
    "session_frequency",
    "order_frequency",
]

CLASSIFIER_FILES = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "SVM": "svm_model.pkl",
    "Ridge Classifier": "ridge_classifier_model.pkl",
}
REGRESSOR_FILE = "linear_regression_model.pkl"


@st.cache_resource
def load_models():
    loaded = {}
    missing = []
    for display_name, filename in CLASSIFIER_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            missing.append(str(path))
        else:
            with path.open("rb") as f:
                loaded[display_name] = pickle.load(f)

    reg_path = MODEL_DIR / REGRESSOR_FILE
    if not reg_path.exists():
        missing.append(str(reg_path))
    else:
        with reg_path.open("rb") as f:
            loaded["Linear Regression"] = pickle.load(f)

    if missing:
        raise FileNotFoundError(
            "The following model files are missing:\n" + "\n".join(missing)
        )
    return loaded


@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Put customer_churn_dataset.csv "
            "in the same folder as app.py."
        )
    return pd.read_csv(DATASET_PATH)


def build_model_input(values):
    """Return a one-row DataFrame in the exact feature order used at training."""
    row = {feature: float(values[feature]) for feature in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)


def describe_class(value):
    """Translate the binary target when it is encoded as 0/1."""
    try:
        numeric = int(value)
        if numeric == 1:
            return "Will renew (1)"
        if numeric == 0:
            return "Will not renew (0)"
    except (TypeError, ValueError):
        pass
    return str(value)


def predict_classifier(model, X):
    pred = model.predict(X)[0]
    result = {
        "prediction": pred,
        "label": describe_class(pred),
        "probability": None,
        "score_note": None,
    }

    # Only show a probability when the model genuinely exposes predict_proba.
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)[0]
            classes = getattr(model, "classes_", np.arange(len(probabilities)))
            positive_index = next(
                (i for i, c in enumerate(classes) if str(c) == "1"),
                len(probabilities) - 1,
            )
            result["probability"] = float(probabilities[positive_index])
        except Exception as exc:
            result["score_note"] = f"Probability unavailable: {exc}"
    elif hasattr(model, "decision_function"):
        try:
            score = np.asarray(model.decision_function(X)).ravel()[0]
            result["score_note"] = (
                f"Decision score: {score:.4f} (not a probability)"
            )
        except Exception:
            result["score_note"] = "No probability output available."

    return result


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


st.title("📊 Customer Auto-Renewal Prediction")
st.caption(
    "Compare five trained classification models and view the separate "
    "Linear Regression numeric output."
)

try:
    models = load_models()
    customers = load_dataset()
except Exception as exc:
    st.error(f"Could not load the application files: {exc}")
    st.info(
        "Check that the six .pkl files are inside the 'models' folder and "
        "the CSV is beside app.py."
    )
    st.stop()

required_dataset_columns = [
    "customer_id", "duration_days", "completed_deliveries",
    "delayed_deliveries", "total_contacts", "complaints",
    "complaint_ratio", "total_web_visits", "total_orders",
    "session_frequency", "order_frequency",
]
missing_columns = [c for c in required_dataset_columns if c not in customers.columns]
if missing_columns:
    st.error("Dataset is missing required columns: " + ", ".join(missing_columns))
    st.stop()

# Check saved model feature names against the app's feature order.
for model_name, model in models.items():
    saved_features = getattr(model, "feature_names_in_", None)
    if saved_features is not None and list(saved_features) != FEATURES:
        st.error(
            f"{model_name} expects a different feature order/naming: "
            f"{list(saved_features)}"
        )
        st.stop()
    n_features = getattr(model, "n_features_in_", len(FEATURES))
    if n_features != len(FEATURES):
        st.error(f"{model_name} expects {n_features} features, not {len(FEATURES)}.")
        st.stop()

with st.sidebar:
    st.header("Customer input")
    input_mode = st.radio(
        "Choose input method",
        ["Select existing customer", "Manual input"],
        index=0,
    )
    st.divider()
    st.caption("The application uses the existing saved models. It does not retrain them.")

if input_mode == "Select existing customer":
    customer_ids = customers["customer_id"].astype(str).tolist()
    selected_id = st.selectbox("Select Customer ID", customer_ids)
    selected_row = customers.loc[
        customers["customer_id"].astype(str) == selected_id
    ].iloc[0]

    total_orders = safe_float(selected_row["total_orders"])
    completed = safe_float(selected_row["completed_deliveries"])
    # Assumption: delivery_ratio = completed_deliveries / total_orders.
    delivery_ratio = completed / total_orders if total_orders > 0 else 0.0

    values = {
        "duration_days": safe_float(selected_row["duration_days"]),
        "completed_deliveries": completed,
        "delayed_deliveries": safe_float(selected_row["delayed_deliveries"]),
        "delivery_ratio": delivery_ratio,
        "total_contacts": safe_float(selected_row["total_contacts"]),
        "complaints": safe_float(selected_row["complaints"]),
        "complaint_ratio": safe_float(selected_row["complaint_ratio"]),
        "total_web_visits": safe_float(selected_row["total_web_visits"]),
        "orders": total_orders,  # CSV column total_orders maps to model feature orders.
        "session_frequency": safe_float(selected_row["session_frequency"]),
        "order_frequency": safe_float(selected_row["order_frequency"]),
    }

    with st.expander("View selected customer's input features"):
        st.dataframe(
            pd.DataFrame([values], columns=FEATURES),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.subheader("Enter customer details")
    st.info(
        "Enter the 10 source values below. delivery_ratio is calculated "
        "automatically as completed_deliveries / total_orders."
    )

    numeric_defaults = {
        "duration_days": float(customers["duration_days"].median()),
        "completed_deliveries": float(customers["completed_deliveries"].median()),
        "delayed_deliveries": float(customers["delayed_deliveries"].median()),
        "total_contacts": float(customers["total_contacts"].median()),
        "complaints": float(customers["complaints"].median()),
        "complaint_ratio": float(customers["complaint_ratio"].median()),
        "total_web_visits": float(customers["total_web_visits"].median()),
        "total_orders": float(customers["total_orders"].median()),
        "session_frequency": float(customers["session_frequency"].median()),
        "order_frequency": float(customers["order_frequency"].median()),
    }

    with st.form("manual_customer_form"):
        c1, c2 = st.columns(2)
        with c1:
            duration_days = st.number_input(
                "Duration days", min_value=0.0,
                value=numeric_defaults["duration_days"], step=1.0
            )
            total_orders = st.number_input(
                "Total orders", min_value=0.0,
                value=numeric_defaults["total_orders"], step=1.0
            )
            completed_deliveries = st.number_input(
                "Completed deliveries", min_value=0.0,
                value=numeric_defaults["completed_deliveries"], step=1.0
            )
            delayed_deliveries = st.number_input(
                "Delayed deliveries", min_value=0.0,
                value=numeric_defaults["delayed_deliveries"], step=1.0
            )
            total_contacts = st.number_input(
                "Total contacts", min_value=0.0,
                value=numeric_defaults["total_contacts"], step=1.0
            )
        with c2:
            complaints = st.number_input(
                "Complaints", min_value=0.0,
                value=numeric_defaults["complaints"], step=1.0
            )
            complaint_ratio = st.number_input(
                "Complaint ratio", min_value=0.0, max_value=1.0,
                value=min(1.0, max(0.0, numeric_defaults["complaint_ratio"])),
                step=0.01, format="%.4f"
            )
            total_web_visits = st.number_input(
                "Total web visits", min_value=0.0,
                value=numeric_defaults["total_web_visits"], step=1.0
            )
            session_frequency = st.number_input(
                "Session frequency", min_value=0.0,
                value=numeric_defaults["session_frequency"], step=0.1
            )
            order_frequency = st.number_input(
                "Order frequency", min_value=0.0,
                value=numeric_defaults["order_frequency"], step=0.1
            )

        submitted = st.form_submit_button(
            "Predict customer auto-renewal", type="primary", use_container_width=True
        )

    if not submitted:
        st.stop()

    delivery_ratio = (
        completed_deliveries / total_orders if total_orders > 0 else 0.0
    )
    values = {
        "duration_days": duration_days,
        "completed_deliveries": completed_deliveries,
        "delayed_deliveries": delayed_deliveries,
        "delivery_ratio": delivery_ratio,
        "total_contacts": total_contacts,
        "complaints": complaints,
        "complaint_ratio": complaint_ratio,
        "total_web_visits": total_web_visits,
        "orders": total_orders,
        "session_frequency": session_frequency,
        "order_frequency": order_frequency,
    }

if input_mode == "Select existing customer":
    run_prediction = st.button(
        "Run all model predictions", type="primary", use_container_width=True
    )
else:
    run_prediction = True

if run_prediction:
    if values["orders"] <= 0:
        st.error("Total orders must be greater than zero for a meaningful delivery ratio.")
        st.stop()

    X = build_model_input(values)

    st.subheader("Classification results")
    results = {}
    for model_name in CLASSIFIER_FILES:
        try:
            results[model_name] = predict_classifier(models[model_name], X)
        except Exception as exc:
            results[model_name] = {
                "prediction": None,
                "label": "Prediction failed",
                "probability": None,
                "score_note": str(exc),
            }

    cards = st.columns(5)
    for col, (name, result) in zip(cards, results.items()):
        with col:
            st.metric(name, result["label"])
            if result["probability"] is not None:
                st.caption(
                    f"Estimated P(renew=1): {result['probability']:.1%}"
                )
            elif result["score_note"]:
                st.caption(result["score_note"])

    table_rows = []
    for name, result in results.items():
        table_rows.append({
            "Model": name,
            "Prediction": result["label"],
            "Renewal probability": (
                f"{result['probability']:.1%}"
                if result["probability"] is not None else "Not available"
            ),
            "Additional information": result["score_note"] or "—",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    valid_predictions = [
        int(r["prediction"]) for r in results.values()
        if r["prediction"] is not None
    ]
    if valid_predictions:
        renew_votes = sum(p == 1 for p in valid_predictions)
        not_renew_votes = sum(p == 0 for p in valid_predictions)
        majority = "Will renew" if renew_votes > not_renew_votes else (
            "Will not renew" if not_renew_votes > renew_votes else "Tie"
        )
        st.subheader("Model agreement")
        a, b, c = st.columns(3)
        a.metric("Renew votes", f"{renew_votes} / {len(valid_predictions)}")
        b.metric("Not-renew votes", f"{not_renew_votes} / {len(valid_predictions)}")
        c.metric("Majority prediction", majority)

    st.divider()
    st.subheader("Linear Regression — separate numeric output")
    try:
        regression_value = float(models["Linear Regression"].predict(X)[0])
        st.metric("Linear Regression output", f"{regression_value:.6f}")
        st.caption(
            "This is a continuous numeric estimate, not a classifier label "
            "or a calibrated probability. Do not interpret it as a renewal "
            "percentage unless the regression target and training method "
            "were specifically designed for that interpretation."
        )
    except Exception as exc:
        st.error(f"Linear Regression prediction failed: {exc}")

    st.divider()
    st.subheader("Prediction input summary")
    st.dataframe(
        pd.DataFrame([values], columns=FEATURES),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = pd.DataFrame([{
        "customer_id": selected_id if input_mode == "Select existing customer" else "Manual input",
        **values,
        **{f"{name}_prediction": result["label"] for name, result in results.items()},
    }]).to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download prediction results (CSV)",
        data=csv_bytes,
        file_name="customer_auto_renewal_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("Important model and data notes"):
    st.markdown(
        """
        - The five classifiers are loaded from the supplied `.pkl` files; no model is retrained.
        - Probability is displayed only for models that implement `predict_proba`.
        - SVM may show a decision score instead. A decision score is **not** a probability.
        - Ridge Classifier may not provide probability estimates.
        - The CSV's `total_orders` column is mapped to the model's `orders` feature.
        - `delivery_ratio` is currently assumed to be `completed_deliveries / total_orders`.
        - The Linear Regression output is shown separately because it is a regressor.
        - Predictions depend on the feature definitions and preprocessing used when the models were trained.
        """
    )
