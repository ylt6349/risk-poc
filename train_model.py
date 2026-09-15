import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

df = pd.read_parquet("data/credit_feature.parquet")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = lgb.LGBMClassifier(
    scale_pos_weight=0.6,
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC: {auc:.4f}")
print(classification_report(y_test, model.predict(X_test)))

model_dir = Path("model")
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "lgb_model.pkl")
print("模型已保存到 model/lgb_model.pkl")
