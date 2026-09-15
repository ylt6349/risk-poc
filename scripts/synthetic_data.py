import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
n = 20000

age = np.random.randint(20, 65, size=n)
income = np.random.lognormal(9.7, 0.5, size=n).astype(int)
credit_history = np.random.poisson(5, size=n)
register_days = np.random.randint(30, 1800, size=n)
debt_ratio = np.random.uniform(0.05, 0.7, size=n)
num_loan = np.random.poisson(2, size=n)
overdue_30 = np.random.binomial(1, 0.22, size=n)

logit = -2.2 + 0.015*age - 0.00003*income + 0.25*num_loan + 1.8*debt_ratio + 1.4*overdue_30
p = 1/(1+np.exp(-logit))
target = np.random.binomial(1, p)

df = pd.DataFrame({
    "age":age,
    "income":income,
    "credit_history":credit_history,
    "register_days":register_days,
    "debt_ratio":debt_ratio,
    "num_loan":num_loan,
    "overdue_30":overdue_30,
    "target":target
})

out_path = Path("data")
out_path.mkdir(exist_ok=True)
df.to_parquet(out_path/"credit_data.parquet", index=False)
print("数据生成完成，保存至 data/credit_data.parquet")
print(df.shape)
print(df["target"].value_counts(normalize=True))
