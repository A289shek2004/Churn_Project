import joblib
pipeline = joblib.load('churn_pipeline.joblib')

input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    # ... all other raw inputs ...
}

import pandas as pd
df = pd.DataFrame([input_dict])
pred = pipeline.predict(df)[0]
