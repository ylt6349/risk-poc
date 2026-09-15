from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="信贷风控预测服务")
model = joblib.load("model/lgb_model.pkl")

class CreditInput(BaseModel):
    age: float
    income: float
    credit_history: float
    register_days: float
    debt_ratio: float
    num_loan: float
    overdue_30d: float
    month_income: float
    debt_income_ratio: float
    loan_per_year: float
    age_group_31_40: float
    age_group_41_50: float
    age_group_51_65: float

@app.post("/predict")
def predict(data: CreditInput):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[0, 1]
    result = "高风险" if prob >= 0.5 else "低风险"
    return {"逾期概率": round(float(prob),4), "风险判定": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8000)
