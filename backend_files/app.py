# Backend REST API for the SuperKart sales forecasting model
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialise the Flask application
superkart_api = Flask("SuperKart Sales Forecaster")
superkart_api.json.sort_keys = False  # keep the row order of the batch predictions

# Load the serialized pipeline (preprocessing + tuned Random Forest) once, when the server starts
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

# Input schema expected by the model
NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
CATEGORICAL_FEATURES = ["Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
                        "Store_Type", "Product_Id_char", "Product_Type_Category"]
FEATURES = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
            "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
            "Store_Age_Years", "Product_Type_Category"]


def prepare_input(df):
    """Validate the incoming data and return a clean dataframe with the model's input columns."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df = df[FEATURES].copy()

    # Categorical fields must be present and non-empty
    for col in CATEGORICAL_FEATURES:
        if df[col].isnull().any() or (df[col].astype(str).str.strip() == "").any():
            raise ValueError(f"'{col}' cannot be empty")
        df[col] = df[col].astype(str).str.strip()

    # Numeric fields must be valid numbers
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_cols = [col for col in NUMERIC_FEATURES if df[col].isnull().any()]
    if invalid_cols:
        raise ValueError(f"Invalid or empty numeric value in: {invalid_cols}")

    # Basic sanity checks on the ranges
    if (df["Product_Weight"] <= 0).any() or (df["Product_MRP"] <= 0).any():
        raise ValueError("Product_Weight and Product_MRP must be greater than 0")
    if ((df["Product_Allocated_Area"] < 0) | (df["Product_Allocated_Area"] > 1)).any():
        raise ValueError("Product_Allocated_Area must be between 0 and 1")
    if (df["Store_Age_Years"] < 0).any():
        raise ValueError("Store_Age_Years cannot be negative")
    return df


# Health-check / welcome route
@superkart_api.get("/")
def home():
    return jsonify({
        "message": "Welcome to the SuperKart Sales Forecasting API!",
        "endpoints": {"online prediction": "POST /v1/predict (JSON)",
                      "batch prediction": "POST /v1/predictbatch (CSV file in form field 'file')"},
    })


# Online (single record) prediction endpoint
@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object with the product and store details"}), 400
        input_df = prepare_input(pd.DataFrame([data]))
        prediction = float(model.predict(input_df)[0])
        return jsonify({"Predicted_Sales": round(prediction, 2)})
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception:
        superkart_api.logger.exception("Prediction failed")
        return jsonify({"error": "Internal server error while generating the prediction"}), 500


# Batch prediction endpoint (CSV file uploaded in the form field 'file')
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "Please upload a CSV file in the form field 'file'"}), 400
        input_df = pd.read_csv(file)
        if input_df.empty:
            return jsonify({"error": "The uploaded CSV file has no rows"}), 400
        input_df = prepare_input(input_df)
        predictions = model.predict(input_df)
        # {row index: predicted sales}
        return jsonify({str(idx): round(float(p), 2) for idx, p in zip(input_df.index, predictions)})
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as err:
        return jsonify({"error": str(err)}), 400
    except Exception:
        superkart_api.logger.exception("Batch prediction failed")
        return jsonify({"error": "Internal server error while generating the predictions"}), 500


# Run the app directly (inside Docker the app is served by gunicorn instead)
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
