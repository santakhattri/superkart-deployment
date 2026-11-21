# SuperKart Sales Forecasting - Model Deployment

A tuned Random Forest model that predicts the total sales revenue of a product in a SuperKart store,
served through a **Flask REST API** (backend) and a **Streamlit app** (frontend), each in its own Docker container.

## Repository structure
```
backend_files/    Flask API: app.py, superkart_model.joblib, requirements.txt, Dockerfile   (port 7860)
frontend_files/   Streamlit UI: app.py, requirements.txt, Dockerfile                        (port 8501)
docker-compose.yml  starts both containers on a shared network
```

## Run the containers

**In a GitHub Codespace** (builds with the host network, because `pip install` inside a container can time out on the Codespaces bridge network):

```bash
docker build --network host -t superkart-backend ./backend_files
docker build --network host -t superkart-frontend ./frontend_files
docker network create superkart-net
docker run -d --name superkart-backend --network superkart-net -p 7860:7860 --restart unless-stopped superkart-backend
docker run -d --name superkart-frontend --network superkart-net -p 8501:8501 --restart unless-stopped \
  -e BACKEND_URL=http://superkart-backend:7860 superkart-frontend
```

**On a machine with normal Docker networking** one command is enough:

```bash
docker compose up --build -d
```

In a GitHub Codespace, open the **Ports** tab and set port **7860** (API) and **8501** (UI) to *Public*.

## API
| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/` | GET | - | health check / welcome message |
| `/v1/predict` | POST | JSON with the 10 input features | `{"Predicted_Sales": <float>}` |
| `/v1/predictbatch` | POST | CSV file in the form field `file` | `{"<row index>": <predicted sales>, ...}` |

Input features: `Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, `Product_MRP`, `Store_Size`,
`Store_Location_City_Type`, `Store_Type`, `Product_Id_char`, `Store_Age_Years`, `Product_Type_Category`.

```bash
curl -X POST http://localhost:7860/v1/predict -H "Content-Type: application/json" -d '{
  "Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar", "Product_Allocated_Area": 0.027,
  "Product_MRP": 117.08, "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
  "Store_Type": "Supermarket Type2", "Product_Id_char": "FD", "Store_Age_Years": 16,
  "Product_Type_Category": "Non Perishables"}'
```
