import pandas as pd
import numpy as np
from ml_models import (
    StandardScalerCustom,
    PowerTransformerCustom,
    train_test_split_custom,
    LinearRegressionCustom,
    RidgeRegressionCustom,
    LassoRegressionCustom,
    ElasticNetRegressionCustom,
    DecisionTreeRegressorCustom,
    RandomForestRegressorCustom,
    ExtraTreesRegressorCustom,
    GradientBoostingRegressorCustom,
    AdaBoostRegressorCustom,
    KNeighborsRegressorCustom,
    SVRCustom
)

# Load data
df = pd.read_csv("insurance_claim_prediction.csv")

# Clean whitespace if any
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

print("Columns:", list(df.columns))
print("Data shape:", df.shape)

# Identify feature types
numeric_cols = [
    'Age', 'Vehicle_Age', 'Vehicle_Value', 'Driving_Experience',
    'Annual_Mileage', 'Accidents_Last_5Yrs', 'Traffic_Violations', 'Credit_Score'
]
categorical_cols = ['Gender', 'Policy_Type']
target_col = 'Claim_Amount'

# Determine skewness of each numeric feature
skew_dict = {}
for col in numeric_cols:
    skewness = df[col].skew()
    skew_dict[col] = skewness
    print(f"Feature: {col}, Skewness: {skewness:.4f}")

# Compute correlation between individual features and target (before preprocessing)
numeric_df = df[numeric_cols + [target_col]].copy()
correlations = numeric_df.corr()[target_col].drop(target_col)
print("\nCorrelation with target variable:")
print(correlations)

# Custom Preprocessor for Features
# "If it skewed preprocess with Power Transformer. Then preprocessor with Standard Scaler. If it is not skewed, preprocess with Standard Scale directly."
processed_features = []
feature_names = []

for col in numeric_cols:
    is_skewed = abs(skew_dict[col]) > 0.5
    col_data = df[[col]].values
    if is_skewed:
        print(f"Feature '{col}' is skewed. Using PowerTransformer + StandardScaler.")
        pt = PowerTransformerCustom()
        col_trans = pt.fit_transform(col_data)
        ss = StandardScalerCustom()
        col_scaled = ss.fit_transform(col_trans)
        processed_features.append(col_scaled)
        feature_names.append(col)
    else:
        print(f"Feature '{col}' is NOT skewed. Using StandardScaler directly.")
        ss = StandardScalerCustom()
        col_scaled = ss.fit_transform(col_data)
        processed_features.append(col_scaled)
        feature_names.append(col)

# Categorical features - One-Hot Encoding Custom
for col in categorical_cols:
    # Get dummies
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(float)
    for d_col in dummies.columns:
        processed_features.append(dummies[[d_col]].values)
        feature_names.append(d_col)

# Combine into a single feature matrix X
X = np.hstack(processed_features)
y = df[target_col].values.reshape(-1, 1)

# Preprocess target variable with StandardScaler
target_scaler = StandardScalerCustom()
y_scaled = target_scaler.fit_transform(y)

print("\nShape of X after preprocessing:", X.shape)
print("Shape of y after preprocessing:", y_scaled.shape)

# Split the data into train test (80:20)
X_train, X_test, y_train, y_test = train_test_split_custom(X, y_scaled, test_size=0.2, random_state=42)

# Verify size
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Build the 11 regressors
regressors = {
    'Linear Regression': LinearRegressionCustom(),
    'Ridge': RidgeRegressionCustom(alpha=1.0),
    'Lasso': LassoRegressionCustom(alpha=0.1),
    'ElasticNet': ElasticNetRegressionCustom(alpha=0.1, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressorCustom(max_depth=3),
    'Random Forest': RandomForestRegressorCustom(n_estimators=10, max_depth=3, random_state=42),
    'Extra Trees': ExtraTreesRegressorCustom(n_estimators=10, max_depth=3, random_state=42),
    'Gradient Boosting': GradientBoostingRegressorCustom(n_estimators=10, learning_rate=0.1, max_depth=2, random_state=42),
    'AdaBoost': AdaBoostRegressorCustom(n_estimators=10, random_state=42),
    'K-Neighbors': KNeighborsRegressorCustom(n_neighbors=2),
    'SVR': SVRCustom(C=1.0, epsilon=0.1, lr=0.01, epochs=1000)
}

results = []

for name, model in regressors.items():
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    
    # Invert scale back to original Claim_Amount
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test_original = target_scaler.inverse_transform(y_test)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test_original - y_pred)**2))
    results.append({'Model': name, 'RMSE': rmse})

results_df = pd.DataFrame(results).sort_values(by='RMSE')
print("\nModel Evaluation (RMSE):")
print(results_df.to_string(index=False))
