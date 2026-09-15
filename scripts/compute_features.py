import pandas as pd
from pathlib import Path

def compute_features(df):
    df["month_income"] = df["income"] / 12
    df["debt_income_ratio"] = df["debt_ratio"]
    df["loan_per_year"] = df["num_loan"] / (df["register_days"]/365 + 1e-6)
    df["age_group"] = pd.cut(df["age"], bins=[20,30,40,50,65], labels=["20-30","31-40","41-50","51-65"])
    df = pd.get_dummies(df, columns=["age_group"])
    return df

if __name__ == "__main__":
    data_path = Path("data/credit_data.parquet")
    df = pd.read_parquet(data_path)
    df_feat = compute_features(df)
    out = Path("data/credit_feature.parquet")
    df_feat.to_parquet(out, index=False)
    print("特征工程完成，保存至", out)
    print(df_feat.shape)
