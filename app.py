# app.py
import streamlit as st
import numpy as np
import joblib
import sklearn
from io import BytesIO
from typing import Tuple, Optional

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(page_title="FinSeg", page_icon="💳", layout="wide")

# --------------------- HELPERS ---------------------
def load_joblib_from_filelike(f):
    """Load a joblib object from a stream/file-like object or path string."""
    if hasattr(f, "read"):
        content = f.read()
        return joblib.load(BytesIO(content))
    else:
        return joblib.load(f)

@st.cache_resource
def load_models_from_disk() -> Tuple[Optional[object], Optional[object], Optional[object], Optional[Exception]]:
    """Try to load model, scaler, encoder from disk (same folder)."""
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, scaler, label_encoder, None
    except Exception as e:
        return None, None, None, e

def model_has_monotonic_attribute(model) -> bool:
    """Check if tree-based estimators inside the model have monotonic_cst attribute.
       This is a heuristic for some sklearn version incompatibilities."""
    try:
        # Ensemble (RandomForest)
        if hasattr(model, "estimators_") and len(model.estimators_) > 0:
            return any(hasattr(est, "monotonic_cst") for est in model.estimators_)
        # Single decision tree
        if hasattr(model, "tree_"):
            return hasattr(model, "monotonic_cst") or hasattr(model, "tree_")
    except Exception:
        pass
    return False

def show_fix_instructions():
    st.error(
        """
        **Compatibility problem detected.**  
        This usually happens when a model `.pkl` was trained with a different scikit-learn version
        than the one running in this app.
        
        **Fix options (choose one):**
        1. **Deploy with the same scikit-learn version** used to train the model (recommended).  
           Example `requirements.txt` (recommended stable combo):  
           ```
           numpy==1.26.4
           pandas==2.2.3
           scikit-learn==1.3.2
           joblib==1.3.2
           streamlit
           ```
        2. **Retrain the model** in an environment matching this app's scikit-learn version, then re-upload/save the .pkl files.
        3. Convert the trained model to ONNX during training and use `onnxruntime` in deployment (advanced).
        
        If you need help, upload your `random_forest_model.pkl`, `scaler.pkl`, and `label_encoder.pkl` below and click "Load uploaded files" — the app will attempt to validate them and show the exact error message.
        """
    )

# --------------------- CSS / Header ---------------------
st.markdown(
    """
    <style>
        .main-title { font-size: 34px; font-weight:700; }
        .prediction-box { padding: 14px; border-radius:10px; background-color:#0b0b0b; color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🗂️ Customer Financial Segmentation System")
st.caption("Enter customer info and get an instant spending-category prediction.")
st.info(f"scikit-learn (runtime) version: {sklearn.__version__}")

# --------------------- LOAD MODELS ---------------------
model, scaler, label_encoder, load_error = load_models_from_disk()

if load_error:
    st.warning("Model files not found or failed to load from disk. You can upload them below.")
    st.write("**Expected filenames**: `random_forest_model.pkl`, `scaler.pkl`, `label_encoder.pkl`")

    col1, col2, col3 = st.columns(3)
    with col1:
        model_file = st.file_uploader("Upload random_forest_model.pkl", type=["pkl"])
    with col2:
        scaler_file = st.file_uploader("Upload scaler.pkl", type=["pkl"])
    with col3:
        encoder_file = st.file_uploader("Upload label_encoder.pkl", type=["pkl"])

    if st.button("Load uploaded files"):
        try:
            if model_file and scaler_file and encoder_file:
                model = load_joblib_from_filelike(model_file)
                scaler = load_joblib_from_filelike(scaler_file)
                label_encoder = load_joblib_from_filelike(encoder_file)
                st.success("Uploaded files loaded successfully.")
            else:
                st.error("Please upload all three files.")
        except Exception as e:
            st.error("Failed to load uploaded files.")
            st.exception(e)
else:
    st.success("Model, scaler, and encoder loaded from disk.")

# Quick model sanity info
if model is not None:
    try:
        st.write("Loaded model type:", type(model))
        # try to print some basic info about the model
        if hasattr(model, "n_estimators"):
            st.write("Model n_estimators:", getattr(model, "n_estimators", "N/A"))
        # check monotonic attribute heuristic
        monotonic_ok = model_has_monotonic_attribute(model)
        st.write("Tree estimators contain `monotonic_cst` attribute (heuristic):", monotonic_ok)
    except Exception as e:
        st.write("Couldn't introspect model:", e)

# --------------------- INPUT SECTION ---------------------
st.subheader("👤 Customer Profile")
colA, colB = st.columns(2)

with colA:
    income = st.number_input("Income", min_value=0.0, format="%.2f")
    expenses = st.number_input("Expenses", min_value=0.0, format="%.2f")
    savings_rate = st.number_input("Savings Rate (e.g., 0.25 for 25%)", min_value=0.0, max_value=2.5, format="%.3f")

with colB:
    credit_cards = st.number_input("Credit Cards Count", min_value=0, max_value=50, step=1)
    credit_utilization = st.number_input("Credit Utilization (%)", min_value=0.0, max_value=100.0, format="%.2f")
    emi_count = st.number_input("Ongoing EMIs", min_value=0, max_value=50, step=1)

online_shopping_spend = st.number_input("Online Shopping Spend", min_value=0.0, format="%.2f")
age = st.number_input("Age", min_value=18, max_value=120, step=1)

predict_btn = st.button("Generate Category")

# --------------------- SAMPLE TEST CASES ---------------------
with st.expander("🔎 Sample test cases (one-click)"):
    st.write("Use these to quickly validate the model. They will fill the inputs below.")
    col_s1, col_s2, col_s3 = st.columns(3)
    if col_s1.button("Saver sample"):
        income, expenses, savings_rate, credit_cards, credit_utilization, emi_count, online_shopping_spend, age = (
            70000.0, 20000.0, 0.45, 1, 12.0, 0, 3000.0, 35
        )
        st.experimental_rerun()
    if col_s2.button("Balanced sample"):
        income, expenses, savings_rate, credit_cards, credit_utilization, emi_count, online_shopping_spend, age = (
            60000.0, 35000.0, 0.20, 2, 35.0, 1, 7000.0, 29
        )
        st.experimental_rerun()
    if col_s3.button("Spender sample"):
        income, expenses, savings_rate, credit_cards, credit_utilization, emi_count, online_shopping_spend, age = (
            85000.0, 70000.0, 0.05, 5, 82.0, 3, 25000.0, 32
        )
        st.experimental_rerun()

# --------------------- PREDICTION ---------------------
if predict_btn:
    # Basic input checks
    if expenses > income:
        st.warning("⚠️ Expenses exceed income — prediction may be inaccurate.")
    if savings_rate > 1.0:
        st.info("💡 Savings rate > 1.0 means saving more than income. Confirm input.")

    if model is None or scaler is None or label_encoder is None:
        st.error("Model/scaler/encoder not loaded. Upload them or place them in the app folder.")
        st.stop()

    # prepare input vector
    try:
        input_data = np.array(
            [[float(income), float(expenses), float(savings_rate), int(credit_cards),
              float(credit_utilization), int(emi_count), float(online_shopping_spend), int(age)]]
        )
    except Exception as e:
        st.error(f"Invalid input types: {e}")
        st.stop()

    # scale & predict with exception handling
    try:
        scaled = scaler.transform(input_data)
    except Exception as e:
        st.error("Scaler.transform failed. Check the scaler and feature order.")
        st.exception(e)
        st.stop()

    try:
        # Final safety: check for known incompatibility
        if not model_has_monotonic_attribute(model) and hasattr(model, "estimators_"):
            # This is a heuristic: if model expects monotonic_cst but not present, warn
            # We'll still attempt to predict, but we catch specific errors below.
            st.warning(
                "Model estimators do not expose `monotonic_cst` attribute which may indicate a compatibility issue. "
                "If prediction fails, follow remediation steps shown below."
            )

        y_pred_enc = model.predict(scaled)
        # restore readable label
        try:
            category = label_encoder.inverse_transform(y_pred_enc)[0]
        except Exception:
            category = label_encoder.inverse_transform([int(y_pred_enc[0])])[0]

        st.markdown(f"<div class='prediction-box'>🎯 <b>Predicted Category:</b> {category}</div>", unsafe_allow_html=True)

        # show probabilities if possible
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled)[0]
            classes = list(label_encoder.classes_)
            prob_pairs = sorted(zip(classes, probs), key=lambda x: -x[1])
            st.write("**Class probabilities:**")
            for name, p in prob_pairs:
                st.write(f"- {name}: {p:.2%}")

    except AttributeError as ae:
        # specifically catch attribute errors like monotonic_cst
        st.error("Prediction failed due to AttributeError during prediction.")
        st.exception(ae)
        show_fix_instructions()
    except Exception as e:
        st.error("Prediction failed with an unexpected error.")
        st.exception(e)
        show_fix_instructions()

else:
    st.info("Fill inputs and click **Generate Category**.")

# --------------------- CATEGORY GUIDE ---------------------
st.markdown("---")
st.subheader("📘 Category Guide")
st.write(
    """
### › **Saver**  
Low spending, higher savings, low credit utilization.

### › **Balanced**  
Moderate spenders with steady saving patterns.

### › **Spender**  
High spending and credit utilization, lifestyle-driven expenses.
"""
)
