# 金融信贷风控端到端 PoC
## 项目简介
本项目为信贷逾期风险预测小型端到端实验项目，基于Python实现从数据生成、探索性数据分析、特征工程、LightGBM模型训练，到FastAPI在线预测服务完整流程。
> 项目为数据分析建模POC，模拟信贷业务场景，用于实习项目演示。

## 技术栈
Python, Pandas, Matplotlib, LightGBM, SHAP, FastAPI, Jupyter Notebook
- 数据：脚本生成模拟信贷样本
- EDA：缺失值、分布分析、相关性可视化
- 建模：LightGBM，处理样本不均衡，特征重要性、SHAP模型可解释性分析
- 部署：FastAPI构建预测接口，提供单样本风险打分

## 项目目录
risk-poc/
├── README.md
├── requirements.txt
├── Dockerfile
├── run_demo.sh
├── decision_thresholds.md
├── notebooks/
│   └── risk_eda_model.ipynb
└── scripts/
├── synthetic_data.py
├── compute_features.py
├── train_model.py
└── serve_model.py
