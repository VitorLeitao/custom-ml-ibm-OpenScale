from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
import joblib
import pandas as pd
import time

app = FastAPI(title="Churn Custom ML Engine")

# --------------------------------------------------
# Modelo
# --------------------------------------------------
model = joblib.load("api/churn_model.pkl")

DEPLOYMENT_ID = "churn-classifier"

LABELS = ["No", "Yes"]

# --------------------------------------------------
# 🔹 Pydantic Schemas (Swagger / OpenAPI)
# --------------------------------------------------

class InputData(BaseModel):
    fields: List[str]
    values: List[List[Any]]


class ScoringRequest(BaseModel):
    input_data: List[InputData]


class Prediction(BaseModel):
    fields: List[str]
    labels: List[str]
    values: List[List[Any]]


class ScoringResponse(BaseModel):
    predictions: List[Prediction]


class DeploymentInfo(BaseModel):
    id: str
    name: str
    scoring_endpoint: str


class DeploymentList(BaseModel):
    deployments: List[DeploymentInfo]

# --------------------------------------------------
# 1️⃣ DISCOVERY ENDPOINT
# --------------------------------------------------
@app.get("/v1/deployments", response_model=DeploymentList)
def list_deployments():
    print(DEPLOYMENT_ID)
    print(f"/v1/deployments/{DEPLOYMENT_ID}/online")
    return {
        "deployments": [
            {
                "id": DEPLOYMENT_ID,
                "name": "Churn Classifier",
                "scoring_endpoint": f"/v1/deployments/{DEPLOYMENT_ID}/online"
            }
        ]
    }

# --------------------------------------------------
# 2️⃣ SCORING ENDPOINT (IBM OpenScale padrão)
# --------------------------------------------------
@app.post(
    "/v1/deployments/{deployment_id}/online",
    response_model=ScoringResponse
)
def score(deployment_id: str, payload: ScoringRequest):

    # ✅ Validação do deployment
    if deployment_id != DEPLOYMENT_ID:
        raise HTTPException(status_code=404, detail="Deployment not found")

    start_time = time.time()

    # Payload IBM padrão
    input_block = payload.input_data[0]
    fields = input_block.fields
    values = input_block.values

    # DataFrame respeitando ordem recebida
    df = pd.DataFrame(values, columns=fields)

    # Predições
    predictions = model.predict(df).astype(int)
    probabilities = model.predict_proba(df)

    response_values = []

    for i in range(len(df)):
        row = [
            v.item() if hasattr(v, "item") else v
            for v in df.iloc[i].values
        ]

        pred_label = LABELS[predictions[i]]
        prob = probabilities[i].tolist()

        response_values.append(row + [pred_label, prob])

    return {
        "predictions": [
            {
                # ✅ fields vindos do request (contrato IBM correto)
                "fields": fields + ["prediction", "probability"],
                "labels": LABELS,
                "values": response_values
            }
        ]
    }

# --------------------------------------------------
# (Opcional) Detalhe do deployment
# --------------------------------------------------
@app.get("/v1/deployments/{deployment_id}")
def get_deployment(deployment_id: str):
    if deployment_id != DEPLOYMENT_ID:
        raise HTTPException(status_code=404, detail="Deployment not found")

    return {
        "id": DEPLOYMENT_ID,
        "name": "Churn Classifier",
        "description": "Customer churn prediction model",
        "model_type": "binary-classification",
        "scoring_endpoint": f"/v1/deployments/{DEPLOYMENT_ID}/online"
    }
