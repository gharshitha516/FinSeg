
import streamlit as st
import numpy as np
import joblib
import sklearn
from io import BytesIO

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="FinSeg",
    page_icon="💳",
    layout="wide",
)

# --------------------- HELPERS ---------------------
def load_joblib_from_filelike(f):
    if hasattr(f, "read"):
        content = f.read()
        return joblib.load(BytesIO(content))
    else:
        return joblib.load(f)

@st.cache_resource
def load_models_from_disk():
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, scaler, label_encoder, None
    except Exception as e:
        return None, None, None, e

# --------------------- UI: Header & CSS ---------------------
st.markdown(
    '''
<style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #ffffff;
    }
    .sub-title {
        font-size: 16px;
        color: #bbbbbb;
        margin-bottom: 5px;
    }
    .section {
        width: 75%;
    }
    .prediction-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #111111;
        border: 1px solid #444444;
        color: #ffffff;
        margin-top: 15px;
        font-size: 18px;
        width: 75%;
    }
</style>
''',
    unsafe_allow_html=True,
)

st.markdown("<h1 class='main-title'> 🗂️ Customer Financial Segmentation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter customer information to generate the predicted financial behavior category.</p>", unsafe_allow_html=True)

st.info(f"scikit-learn version: {sklearn.__version__}")

# --------------------- LOAD MODELS ---------------------
model, scaler, label_encoder, load_error = load_models_from_disk()

if load_error:
    st.warning("Model files not found or failed to load. Upload them below.")

    col_upload1, col_upload2, col_upload3 = st.columns(3)
    with col_upload1:
        model_file = st.file_uploader("Upload random_forest_model.pkl", type=["pkl"])
    with col_upload2:
        scaler_file = st.file_uploader("Upload scaler.pkl", type=["pkl"])
    with col_upload3:
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
            st.error(f"Error loading files: {e}")

else:
    st.success("Model, scaler, and encoder loaded from disk.")

# --------------------- INPUT SECTION ---------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("👤 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    st.write("### 💼 Financial Information")
    income = st.number_input("Income", min_value=0.0, format="%.2f")
    expenses = st.number_input("Expenses", min_value=0.0, format="%.2f")
    savings_rate = st.number_input("Savings Rate", min_value=0.0, max_value=2.5, format="%.3f")

with col2:
    st.write("### 💳 Credit Details")
    credit_cards = st.number_input("Credit Cards Count", min_value=0, max_value=50, step=1)
    credit_utilization = st.number_input("Credit Utilization (%)", min_value=0.0, max_value=100.0, format="%.2f")
    emi_count = st.number_input("Ongoing EMIs", min_value=0, max_value=50, step=1)

st.write("### ✨ Lifestyle")
online_shopping_spend = st.number_input("Online Shopping Spend", min_value=0.0, format="%.2f")
age = st.number_input("Age", min_value=18, max_value=120, step=1)

predict_btn = st.button("Generate Category", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------- PREDICTION LOGIC ---------------------
if predict_btn:
    if expenses > income:
        st.warning("⚠️ Expenses exceed income — prediction may be inaccurate.")
    if savings_rate > 1.0:
        st.info("💡 Savings rate above 1.0 means saving more than income.")

    if model is None or scaler is None or label_encoder is None:
        st.error("Model/scaler/encoder not loaded.")
    else:
        try:
            input_data = np.array([
                [
                    float(income),
                    float(expenses),
                    float(savings_rate),
                    int(credit_cards),
                    float(credit_utilization),
                    int(emi_count),
                    float(online_shopping_spend),
                    int(age),
                ]
            ])

            scaled_input = scaler.transform(input_data)
            pred_encoded = model.predict(scaled_input)
            category = label_encoder.inverse_transform(pred_encoded)[0]

            st.markdown(
                f"<div class='prediction-box'>🎯 <b>Predicted Category:</b> {category}</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("Fill in the details and click **Generate Category**.")

# --------------------- CATEGORY GUIDE ---------------------
if 'category' in locals():
    st.markdown("---")
    st.subheader("📘 Category Guide")
    st.write(
        '''
### › **Saver**
Low-spending individuals who maintain controlled financial habits.

### › **Balanced**
Moderate spenders with stable saving and spending patterns.

### › **Spender**
High expenditure, higher credit use, and lifestyle-driven spending.
'''
    )
