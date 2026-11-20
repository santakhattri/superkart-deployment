# Streamlit frontend for the SuperKart sales forecasting API
import io
import os

import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# The backend URL is injected through an environment variable. Inside the Docker network
# the backend container is reachable by its container name.
# ----------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://superkart-backend:7860").rstrip("/")
PREDICT_URL = f"{BACKEND_URL}/v1/predict"
BATCH_URL = f"{BACKEND_URL}/v1/predictbatch"
REQUEST_TIMEOUT = 60
TRAINING_AVG_SALES = 3464.0  # average Product_Store_Sales_Total in the historical data

PRODUCT_TYPES = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
                 "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household", "Meat",
                 "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]
# Same grouping that was used to train the model (Product_Type -> Product_Type_Category)
PERISHABLE_TYPES = {"Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"}

PRODUCT_FAMILIES = {"FD": "FD - Food", "DR": "DR - Drinks", "NC": "NC - Non-consumables"}

# The four existing SuperKart stores (attributes taken from the historical data; age = 2025 - year established)
STORE_PRESETS = {
    "OUT001 - Supermarket Type1 (High, Tier 2)": {"Store_Size": "High", "Store_Location_City_Type": "Tier 2",
                                                  "Store_Type": "Supermarket Type1", "Store_Age_Years": 38},
    "OUT002 - Food Mart (Small, Tier 3)": {"Store_Size": "Small", "Store_Location_City_Type": "Tier 3",
                                           "Store_Type": "Food Mart", "Store_Age_Years": 27},
    "OUT003 - Departmental Store (Medium, Tier 1)": {"Store_Size": "Medium", "Store_Location_City_Type": "Tier 1",
                                                     "Store_Type": "Departmental Store", "Store_Age_Years": 26},
    "OUT004 - Supermarket Type2 (Medium, Tier 2)": {"Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
                                                    "Store_Type": "Supermarket Type2", "Store_Age_Years": 16},
    "Custom / new store": None,
}
STORE_TYPES = ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart", "Supermarket Type3"]

BATCH_COLUMNS = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
                 "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
                 "Store_Age_Years", "Product_Type_Category"]
TEMPLATE_ROW = {"Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar", "Product_Allocated_Area": 0.027,
                "Product_MRP": 117.08, "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
                "Store_Type": "Supermarket Type2", "Product_Id_char": "FD", "Store_Age_Years": 16,
                "Product_Type_Category": "Non Perishables"}


# ----------------------------------------------------------------------------
# Helper functions (kept free of Streamlit calls so that they are easy to test)
# ----------------------------------------------------------------------------
def product_type_to_category(product_type):
    """Map one of the 16 product types to the Perishables / Non Perishables category used by the model."""
    return "Perishables" if product_type in PERISHABLE_TYPES else "Non Perishables"


def build_payload(weight, sugar, area, mrp, family, product_type, store):
    """Create the JSON payload expected by the /v1/predict endpoint."""
    return {
        "Product_Weight": float(weight),
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": float(area),
        "Product_MRP": float(mrp),
        "Store_Size": store["Store_Size"],
        "Store_Location_City_Type": store["Store_Location_City_Type"],
        "Store_Type": store["Store_Type"],
        "Product_Id_char": family,
        "Store_Age_Years": int(store["Store_Age_Years"]),
        "Product_Type_Category": product_type_to_category(product_type),
    }


def _error_message(response):
    """Extract a readable error message from a failed API response."""
    try:
        return response.json().get("error", f"HTTP {response.status_code}")
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"


def get_prediction(payload):
    """Call the online prediction endpoint. Returns (predicted_sales, error_message)."""
    try:
        response = requests.post(PREDICT_URL, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as err:
        return None, f"Could not reach the prediction service ({type(err).__name__}). Is the backend running?"
    if response.status_code == 200:
        return response.json()["Predicted_Sales"], None
    return None, _error_message(response)


def get_batch_predictions(file_bytes, filename="batch.csv"):
    """Call the batch endpoint. Returns (dict {row index: predicted sales}, error_message)."""
    try:
        response = requests.post(BATCH_URL, files={"file": (filename, file_bytes, "text/csv")},
                                 timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as err:
        return None, f"Could not reach the prediction service ({type(err).__name__}). Is the backend running?"
    if response.status_code == 200:
        return response.json(), None
    return None, _error_message(response)


def attach_predictions(input_df, predictions):
    """Add the predicted sales (returned by the API as {row index: value}) to the uploaded data."""
    result = input_df.copy()
    result["Predicted_Sales"] = [predictions.get(str(i)) for i in input_df.index]
    return result


def backend_is_up():
    """Simple health check of the backend API."""
    try:
        return requests.get(BACKEND_URL + "/", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


# ----------------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="SuperKart Sales Forecaster", page_icon="🛒", layout="wide")
    st.title("🛒 SuperKart Sales Forecaster")
    st.caption("Predict the total sales revenue of a product in a SuperKart store to support "
               "inventory planning and regional sales strategy.")

    with st.sidebar:
        st.header("About")
        st.write("This app sends the product and store details to the SuperKart forecasting API "
                 "(a tuned Random Forest model) and shows the predicted sales revenue.")
        st.write("**Single prediction**: one product-store combination.")
        st.write("**Batch prediction**: upload a CSV to forecast many rows at once.")
        if backend_is_up():
            st.success("Prediction service: online")
        else:
            st.error("Prediction service: not reachable")
        st.caption(f"API: {BACKEND_URL}")

    tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

    # ------------------------------ single prediction ------------------------------
    with tab_single:
        col_product, col_store = st.columns(2)

        with col_product:
            st.subheader("Product details")
            product_type = st.selectbox("Product type", PRODUCT_TYPES, index=PRODUCT_TYPES.index("Frozen Foods"))
            st.caption(f"Treated by the model as: **{product_type_to_category(product_type)}**")
            family = st.selectbox("Product family", list(PRODUCT_FAMILIES), format_func=PRODUCT_FAMILIES.get)
            weight = st.number_input("Product weight", min_value=0.5, max_value=50.0, value=12.66, step=0.01)
            sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
            area = st.number_input("Allocated display area (share of total display area, 0 to 1)",
                                   min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f")
            mrp = st.number_input("Maximum retail price (MRP)", min_value=1.0, max_value=1000.0,
                                  value=117.08, step=0.01)

        with col_store:
            st.subheader("Store details")
            store_label = st.selectbox("Store", list(STORE_PRESETS), index=3)
            store = STORE_PRESETS[store_label]
            if store is None:
                store = {
                    "Store_Size": st.selectbox("Store size", ["Small", "Medium", "High"], index=1),
                    "Store_Location_City_Type": st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1),
                    "Store_Type": st.selectbox("Store type", STORE_TYPES),
                    "Store_Age_Years": st.number_input("Store age (years)", min_value=0, max_value=100, value=10),
                }
                if store["Store_Type"] == "Supermarket Type3":
                    st.warning("This store type was not present in the training data, so the prediction "
                               "will be less reliable.")
            else:
                st.info(f"Size: **{store['Store_Size']}**  |  City: **{store['Store_Location_City_Type']}**  |  "
                        f"Type: **{store['Store_Type']}**  |  Age: **{store['Store_Age_Years']} years**")

        if st.button("Predict sales", type="primary"):
            payload = build_payload(weight, sugar, area, mrp, family, product_type, store)
            with st.spinner("Contacting the prediction service..."):
                prediction, error = get_prediction(payload)
            st.session_state["single_result"] = (payload, prediction, error)

        if "single_result" in st.session_state:
            payload, prediction, error = st.session_state["single_result"]
            if error:
                st.error(f"Prediction failed: {error}")
            else:
                st.metric("Predicted sales (total revenue for this product in this store)",
                          f"{prediction:,.2f}",
                          delta=f"{prediction - TRAINING_AVG_SALES:+,.2f} vs the average of {TRAINING_AVG_SALES:,.0f}")
                with st.expander("Request sent to the API"):
                    st.json(payload)

    # ------------------------------ batch prediction ------------------------------
    with tab_batch:
        st.subheader("Batch prediction from a CSV file")
        st.write("The file must contain these columns: " + ", ".join(f"`{c}`" for c in BATCH_COLUMNS))
        st.download_button("Download a CSV template", pd.DataFrame([TEMPLATE_ROW]).to_csv(index=False),
                           file_name="superkart_batch_template.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            try:
                batch_df = pd.read_csv(io.BytesIO(file_bytes))
            except Exception as err:
                batch_df = None
                st.error(f"The file could not be read as a CSV: {err}")

            if batch_df is not None:
                st.write(f"Preview ({len(batch_df)} rows):")
                st.dataframe(batch_df.head(10))
                if st.button("Predict sales for all rows", type="primary"):
                    with st.spinner("Generating predictions..."):
                        predictions, error = get_batch_predictions(file_bytes, uploaded.name)
                    if error:
                        st.session_state["batch_result"] = (None, error)
                    else:
                        st.session_state["batch_result"] = (attach_predictions(batch_df, predictions), None)

        if "batch_result" in st.session_state:
            result_df, error = st.session_state["batch_result"]
            if error:
                st.error(f"Batch prediction failed: {error}")
            else:
                st.success("Predictions generated successfully")
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows predicted", f"{len(result_df):,}")
                m2.metric("Total predicted sales", f"{result_df['Predicted_Sales'].sum():,.2f}")
                m3.metric("Average predicted sales", f"{result_df['Predicted_Sales'].mean():,.2f}")
                st.dataframe(result_df)
                st.bar_chart(result_df["Predicted_Sales"])
                st.download_button("Download predictions as CSV", result_df.to_csv(index=False),
                                   file_name="superkart_predictions.csv", mime="text/csv")


main()
