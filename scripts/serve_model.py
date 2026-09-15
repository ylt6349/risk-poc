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
    age_group_18_30: float
    age_group_41_50: float
    age_group_51_65: float

@app.post("/predict")
def predict(data: CreditInput):
    df = pd.DataFrame([data.dict()])
    # 强制对齐模型训练时特征顺序，解决LightGBM顺序报错
    feature_order = [
        'age',
        'income',
        'credit_history',
        'register_days',
        'debt_ratio',
        'num_loan',
        'overdue_30d',
        'month_income',
        'debt_income_ratio',
        'loan_per_year',
        'age_group_18_30',
        'age_group_31_40',
        'age_group_41_50',
        'age_group_51_65'
    ]
    df = df[feature_order]
    prob = model.predict_proba(df)[0, 1]
    result = "高风险" if prob >= 0.5 else "低风险"    
    return {"逾期概率": round(float(prob),4), "风险判定": result}

if __name__ == "__main__":    
    import uvicorn
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8000)
