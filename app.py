import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
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

# Set page config
st.set_page_config(
    page_title="Insurance Claim Predictor & Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define custom styling for premium looks
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 3rem !important;
        font-weight: 700;
        background: linear-gradient(90deg, #1A365D 0%, #3182CE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #718096;
        margin-bottom: 2rem;
    }
    
    /* Card design */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2B6CB0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #4A5568;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }

    .prediction-box {
        background: linear-gradient(135deg, #2B6CB0 0%, #1A365D 100%);
        color: white;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 10px 25px rgba(43, 108, 176, 0.35);
    }
    
    .prediction-title {
        font-size: 1.2rem;
        font-weight: 600;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .prediction-val {
        font-size: 3.5rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("insurance_claim_prediction.csv")
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

df = load_data()

# ----------------- APP LAYOUT -----------------
st.markdown('<div class="main-title">🛡️ Insurance Claim Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced analytical dashboard & model playground for claim cost prediction.</div>', unsafe_allow_html=True)

# Define column names
numeric_cols = [
    'Age', 'Vehicle_Age', 'Vehicle_Value', 'Driving_Experience',
    'Annual_Mileage', 'Accidents_Last_5Yrs', 'Traffic_Violations', 'Credit_Score'
]
categorical_cols = ['Gender', 'Policy_Type']
target_col = 'Claim_Amount'

# Split raw X and y first (80:20), then fit preprocessing on the training data only.
X_raw = df[numeric_cols + categorical_cols].copy()
y_raw = df[target_col].values.reshape(-1, 1)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split_custom(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# Compute feature skewness from the training split for preprocessing selection.
skew_dict = {col: X_train_raw[col].skew() for col in numeric_cols}
fitted_pt_scalers = {}

for col in numeric_cols:
    is_skewed = abs(skew_dict[col]) > 0.5
    train_col = X_train_raw[[col]].values
    if is_skewed:
        pt = PowerTransformerCustom().fit(train_col)
        train_trans = pt.transform(train_col)
        ss = StandardScalerCustom().fit(train_trans)
        fitted_pt_scalers[col] = (pt, ss, True)
    else:
        ss = StandardScalerCustom().fit(train_col)
        fitted_pt_scalers[col] = (None, ss, False)

# Keep training categories as the one-hot schema used by train, test, and prediction inputs.
cat_dummies_columns = {}
for col in categorical_cols:
    dummies = pd.get_dummies(X_train_raw[col], prefix=col, drop_first=False).astype(float)
    cat_dummies_columns[col] = list(dummies.columns)

def transform_features(raw_features):
    processed_features = []

    for col in numeric_cols:
        col_data = raw_features[[col]].values
        pt, ss, is_skewed = fitted_pt_scalers[col]
        if is_skewed:
            col_data = pt.transform(col_data)
        processed_features.append(ss.transform(col_data))

    for col in categorical_cols:
        dummies = pd.get_dummies(raw_features[col], prefix=col, drop_first=False).astype(float)
        dummies = dummies.reindex(columns=cat_dummies_columns[col], fill_value=0.0)
        processed_features.append(dummies.values)

    return np.hstack(processed_features)

X_train = transform_features(X_train_raw)
X_test = transform_features(X_test_raw)

# Preprocess the target variable with StandardScaler learned on the training target.
target_scaler = StandardScalerCustom().fit(y_train_raw)
y_train = target_scaler.transform(y_train_raw)
y_test = target_scaler.transform(y_test_raw)

# Build and train all 11 regressors
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

trained_models = {}
results = []

for name, model in regressors.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test_original = target_scaler.inverse_transform(y_test)
    
    rmse = np.sqrt(np.mean((y_test_original - y_pred)**2))
    results.append({'Model': name, 'RMSE': rmse})

results_df = pd.DataFrame(results).sort_values(by='RMSE')
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

# Helper to process single user input row
def preprocess_user_input(user_input_dict):
    row_features = []
    
    # Preprocess numeric features
    for col in numeric_cols:
        val = np.array([[float(user_input_dict[col])]])
        pt, ss, is_skewed = fitted_pt_scalers[col]
        if is_skewed:
            val_trans = pt.transform(val)
            val_scaled = ss.transform(val_trans)
            row_features.append(val_scaled[0, 0])
        else:
            val_scaled = ss.transform(val)
            row_features.append(val_scaled[0, 0])
            
    # Preprocess categorical features
    for col in categorical_cols:
        val_cat = user_input_dict[col]
        for d_col in cat_dummies_columns[col]:
            category_name = d_col.split(col + "_")[1]
            if val_cat.lower() == category_name.lower():
                row_features.append(1.0)
            else:
                row_features.append(0.0)
                
    return np.array(row_features).reshape(1, -1)

# Sidebar Info
st.sidebar.markdown("### Model Details")
st.sidebar.markdown(f"**Best Model**: `{best_model_name}`")
st.sidebar.markdown(f"**Best RMSE**: `${results_df.iloc[0]['RMSE']:,.2f}`")
st.sidebar.markdown("---")
st.sidebar.write("Ensure your inputs are entered inside the **Prediction Playground** tab to perform predictions.")

# Main Dashboard Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction Playground", 
    "📊 Feature Exploration & Skewness", 
    "📈 Correlation Analysis", 
    "🤖 Model Comparison & Leaderboard"
])

with tab1:
    st.write("### Interactive Claim Prediction Portal")
    st.write("Select an input mode below to enter feature values. The dashboard will process the variables (matching the exact pipeline) and run prediction using the best model (`" + best_model_name + "`).")

    input_mode = st.radio("Input Method:", ["Interactive Form Inputs", "Raw JSON Copy-Paste"], horizontal=True)

    user_inputs = {}
    is_valid = True

    if input_mode == "Interactive Form Inputs":
        col_form_l, col_form_r = st.columns(2)
        with col_form_l:
            user_inputs['Age'] = st.number_input("Age", min_value=16, max_value=100, value=35, key="form_age")
            user_inputs['Gender'] = st.selectbox("Gender", options=sorted(df['Gender'].unique()), key="form_gender")
            user_inputs['Vehicle_Age'] = st.number_input("Vehicle Age (Years)", min_value=0, max_value=50, value=5, key="form_v_age")
            user_inputs['Vehicle_Value'] = st.number_input("Vehicle Value ($)", min_value=1000, max_value=200000, value=30000, step=1000, key="form_v_val")
            user_inputs['Driving_Experience'] = st.number_input("Driving Experience (Years)", min_value=0, max_value=80, value=12, key="form_exp")
        
        with col_form_r:
            user_inputs['Annual_Mileage'] = st.number_input("Annual Mileage (Miles)", min_value=100, max_value=100000, value=12000, step=500, key="form_mileage")
            user_inputs['Accidents_Last_5Yrs'] = st.number_input("Accidents in Last 5 Years", min_value=0, max_value=20, value=1, key="form_acc")
            user_inputs['Traffic_Violations'] = st.number_input("Traffic Violations", min_value=0, max_value=30, value=0, key="form_viol")
            user_inputs['Credit_Score'] = st.slider("Credit Score", min_value=300, max_value=900, value=720, key="form_credit")
            user_inputs['Policy_Type'] = st.selectbox("Policy Type", options=sorted(df['Policy_Type'].unique()), key="form_policy")

    else:
        st.write("Paste a JSON document containing feature values:")
        default_json = {
            "Age": 30,
            "Gender": "Male",
            "Vehicle_Age": 5,
            "Vehicle_Value": 25000,
            "Driving_Experience": 10,
            "Annual_Mileage": 15000,
            "Accidents_Last_5Yrs": 1,
            "Traffic_Violations": 1,
            "Credit_Score": 700,
            "Policy_Type": "Standard"
        }
        json_str = st.text_area("JSON Input:", value=json.dumps(default_json, indent=4), height=250)
        try:
            user_inputs = json.loads(json_str)
            # Verify required keys
            required_keys = numeric_cols + categorical_cols
            missing_keys = [k for k in required_keys if k not in user_inputs]
            if len(missing_keys) > 0:
                st.error(f"Missing keys in JSON: {missing_keys}")
                is_valid = False
        except Exception as e:
            st.error(f"Invalid JSON Format: {e}")
            is_valid = False

    if is_valid:
        # Predict based on inputs
        user_row_preprocessed = preprocess_user_input(user_inputs)
        prediction_scaled = best_model.predict(user_row_preprocessed).reshape(-1, 1)
        prediction_val = target_scaler.inverse_transform(prediction_scaled)[0, 0]
        if prediction_val < 0:
            prediction_val = 0.0

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-title">Estimated Claim Amount</div>
                <div class="prediction-val">${prediction_val:,.2f}</div>
                <div style="font-size: 0.95rem; margin-top: 0.5rem; opacity: 0.85;">
                    Predicted using <b>{best_model_name}</b> (RMSE: ${results_df.iloc[0]['RMSE']:,.2f})
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with tab2:
    st.write("### Individual Feature Analysis and Preprocessing Selection")
    st.write("Each numerical feature's distribution is visualized below along with its skewness score. If skewness is high ($|skew| > 0.5$), the feature is normalized using a Power Transformer (Yeo-Johnson) followed by Standard Scaling. Otherwise, it is scaled directly.")

    # Create grids for plots
    num_plots = len(numeric_cols)
    cols_per_row = 4
    for i in range(0, num_plots, cols_per_row):
        row_cols = st.columns(cols_per_row)
        for idx, col_name in enumerate(numeric_cols[i:i+cols_per_row]):
            with row_cols[idx]:
                skew_val = skew_dict[col_name]
                is_skewed = abs(skew_val) > 0.5
                
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(df[col_name], kde=True, color='#2B6CB0', ax=ax)
                ax.set_title(f"{col_name}\n(Skewness: {skew_val:.3f})", fontsize=10, fontweight='bold')
                ax.set_xlabel("")
                ax.set_ylabel("")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Preprocessing pill/tag
                if is_skewed:
                    st.markdown(f"🟡 **Skewed** ($|skew| > 0.5$)<br>➔ `PowerTransformer` + `StandardScaler`", unsafe_allow_html=True)
                else:
                    st.markdown(f"🟢 **Not Skewed** ($|skew| \\\\le 0.5$)<br>➔ `StandardScaler` directly", unsafe_allow_html=True)
                st.write("---")

with tab3:
    st.write("### Feature Correlations with Target Variable (`Claim_Amount`)")
    
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        # Calculate Pearson correlations for numeric and one-hot encoded categorical features.
        corr_source = pd.concat(
            [
                df[numeric_cols],
                pd.get_dummies(df[categorical_cols], prefix=categorical_cols, drop_first=False).astype(float),
                df[[target_col]]
            ],
            axis=1
        )
        corr_df = corr_source.corr()
        target_corr = corr_df[[target_col]].drop(target_col).sort_values(by=target_col, ascending=False)
        target_corr.columns = ["Correlation Value"]
        
        st.write("#### Correlation Coefficients")
        st.dataframe(
            target_corr.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1),
            width='stretch'
        )
        
    with col_right:
        st.write("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5, ax=ax)
        plt.title("Correlation Matrix Heatmap", fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

with tab4:
    st.write("### Regressor Leaderboard (RMSE comparison)")
    st.write("All 11 algorithms are trained on 80% of the dataset and evaluated on the remaining 20%. Root Mean Squared Error (RMSE) is calculated on the original target variable scale.")
    
    col_lead_left, col_lead_right = st.columns([1.2, 1])
    
    with col_lead_left:
        # Highlight best model
        st.write("#### Leaderboard Table")
        
        def highlight_best(s):
            is_min = s == s.min()
            return ['background-color: #C6F6D5; font-weight: bold;' if v else '' for v in is_min]
            
        st.dataframe(
            results_df.style.apply(highlight_best, subset=['RMSE']).format({'RMSE': '{:,.2f}'}),
            width='stretch',
            hide_index=True
        )
        
    with col_lead_right:
        st.write("#### Model Performance Chart")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        # Plot horizontal bar chart
        sns.barplot(data=results_df, y='Model', x='RMSE', hue='Model', palette='viridis_r', legend=False, ax=ax)
        ax.set_title("RMSE Comparison (Lower is Better)", fontsize=11, fontweight='bold')
        ax.set_xlabel("RMSE ($)")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        
    # Metrics breakdown
    st.write("---")
    st.write("### Model Insight Summary")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{best_model_name}</div>
                <div class="metric-label">Best Performing Model</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">${results_df.iloc[0]['RMSE']:,.2f}</div>
                <div class="metric-label">Best Model RMSE</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with m_col3:
        # Get baseline Linear Regression RMSE
        lr_rmse = results_df[results_df['Model'] == 'Linear Regression'].iloc[0]['RMSE']
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">${lr_rmse:,.2f}</div>
                <div class="metric-label">Linear Regression RMSE (Baseline)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
