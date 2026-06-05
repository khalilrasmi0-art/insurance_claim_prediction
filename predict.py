import sys
import json
import numpy as np
import pandas as pd
from ml_models import (
    StandardScalerCustom,
    PowerTransformerCustom,
    train_test_split_custom,
    LinearRegressionCustom
)

# Load data to fit pipeline state
df = pd.read_csv("insurance_claim_prediction.csv")
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

numeric_cols = [
    'Age', 'Vehicle_Age', 'Vehicle_Value', 'Driving_Experience',
    'Annual_Mileage', 'Accidents_Last_5Yrs', 'Traffic_Violations', 'Credit_Score'
]
categorical_cols = ['Gender', 'Policy_Type']
target_col = 'Claim_Amount'

# Determine skewness of each numeric feature
skew_dict = {col: df[col].skew() for col in numeric_cols}

# Fit transformers
processed_features = []
fitted_pt_scalers = {}

for col in numeric_cols:
    is_skewed = abs(skew_dict[col]) > 0.5
    col_data = df[[col]].values
    if is_skewed:
        pt = PowerTransformerCustom().fit(col_data)
        col_trans = pt.transform(col_data)
        ss = StandardScalerCustom().fit(col_trans)
        col_scaled = ss.transform(col_trans)
        processed_features.append(col_scaled)
        fitted_pt_scalers[col] = (pt, ss, True)
    else:
        ss = StandardScalerCustom().fit(col_data)
        col_scaled = ss.transform(col_data)
        processed_features.append(col_scaled)
        fitted_pt_scalers[col] = (None, ss, False)

cat_dummies_columns = {}
for col in categorical_cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(float)
    cat_dummies_columns[col] = list(dummies.columns)
    for d_col in dummies.columns:
        processed_features.append(dummies[[d_col]].values)

X_processed = np.hstack(processed_features)
y_raw = df[target_col].values.reshape(-1, 1)

target_scaler = StandardScalerCustom().fit(y_raw)
y_processed = target_scaler.transform(y_raw)

# Train the best model (Linear Regression)
X_train, X_test, y_train, y_test = train_test_split_custom(X_processed, y_processed, test_size=0.2, random_state=42)
best_model = LinearRegressionCustom()
best_model.fit(X_train, y_train)

def predict_single(input_dict):
    row_features = []
    
    # Numeric
    for col in numeric_cols:
        val = np.array([[float(input_dict[col])]])
        pt, ss, is_skewed = fitted_pt_scalers[col]
        if is_skewed:
            val_trans = pt.transform(val)
            val_scaled = ss.transform(val_trans)
            row_features.append(val_scaled[0, 0])
        else:
            val_scaled = ss.transform(val)
            row_features.append(val_scaled[0, 0])
            
    # Categorical
    for col in categorical_cols:
        val_cat = input_dict[col].strip()
        for d_col in cat_dummies_columns[col]:
            category_name = d_col.split(col + "_")[1]
            if val_cat.lower() == category_name.lower():
                row_features.append(1.0)
            else:
                row_features.append(0.0)
                
    row_arr = np.array(row_features).reshape(1, -1)
    pred_scaled = best_model.predict(row_arr).reshape(-1, 1)
    pred_val = target_scaler.inverse_transform(pred_scaled)[0, 0]
    return max(0.0, pred_val)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Read from json string argument
        try:
            user_data = json.loads(sys.argv[1])
            res = predict_single(user_data)
            print(f"PREDICTION: {res:.2f}")
        except Exception as e:
            print(f"ERROR: {str(e)}")
    else:
        print("Provide input as a JSON string. Example: python predict.py \"{\\\"Age\\\": 30, \\\"Gender\\\": \\\"Male\\\", \\\"Vehicle_Age\\\": 5, \\\"Vehicle_Value\\\": 25000, \\\"Driving_Experience\\\": 10, \\\"Annual_Mileage\\\": 15000, \\\"Accidents_Last_5Yrs\\\": 1, \\\"Traffic_Violations\\\": 1, \\\"Credit_Score\\\": 700, \\\"Policy_Type\\\": \\\"Standard\\\"}\"")
