Overview:-
This project focuses on detecting fraudulent transactions using supervised machine learning techniques.
The goal is to build a reliable, interpretable, and production-ready fraud detection pipeline that can distinguish between legitimate and fraudulent transactions while handling severe class imbalance.
The project is designed to demonstrate end-to-end machine learning skills, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment-readiness.

Streamlit app:-
https://frauddetection-mrunalasutkar.streamlit.app/

Tech Stack:-
Language: Python
Libraries:
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Joblib

Folder Structure:-
fraud_detection/
│
├── data/
│ └── creditcard.csv
│
├── notebooks/
│ └── data_cleaning.ipynb
| └── modeling_baseline.ipynb
| └── modeling_advanced.ipynb
| └── model_test.ipynb
| └── deployment.ipynb
│
├── models/
│ └── amount_scaler.pkl
| └── xgb_model.pkl
│
├── app
| └── app.py 
├── requirements.txt
└── README.md

