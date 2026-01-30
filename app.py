import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Lung Cancer Risk AI", layout="centered")

st.title("🫁 Lung Cancer Risk Prediction AI")
st.write("Enter patient details to check lung cancer risk")

# ================= DATA LOADING =================
@st.cache_data
def load_data():
    df = pd.read_csv("lung_cancer.csv")
    df.drop([
        'education_years','income_level','pack_years',
        'occupational_exposure','radon_exposure','copd',
        'fev1_x10','crp_level'
    ], axis=1, inplace=True)
    return df

df = load_data()

X = df.drop('lung_cancer_risk', axis=1)
y = df['lung_cancer_risk']
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODEL =================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ================= USER INPUT =================
st.subheader("🧑 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["male", "female"])
    smoker = st.selectbox("Do you smoke?", ["yes", "no"])
    smoking_years = st.number_input("Smoking Years", 0, 60, 0)
    cigarettes = st.number_input("Cigarettes per day", 0, 50, 0)
    passive = st.selectbox("Passive Smoking?", ["yes", "no"])
    bmi = st.number_input("BMI", 10, 50, 22)

with col2:
    air = st.number_input("Air Pollution Index", 0, 500, 80)
    family = st.selectbox("Family History of Cancer?", ["yes", "no"])
    asthma = st.selectbox("Asthma?", ["yes", "no"])
    tb = st.selectbox("Previous TB?", ["yes", "no"])
    cough = st.selectbox("Chronic Cough?", ["yes", "no"])
    chest_pain = st.selectbox("Chest Pain?", ["yes", "no"])
    oxygen_saturation = st.number_input("Oxygen Saturation", 50, 100, 98)

xray_abnormal = st.selectbox("X-Ray Abnormal?", ["yes", "no"])
breath = st.selectbox("Shortness of Breath?", ["yes", "no"])
fatigue = st.selectbox("Fatigue?", ["yes", "no"])
exercise = st.number_input("Exercise Hours / Week", 0, 20, 2)
diet = st.slider("Diet Quality (1-5)", 1, 5, 3)
alcohol = st.number_input("Alcohol Units / Week", 0, 50, 0)
healthcare = st.slider("Healthcare Access (1-5)", 1, 5, 3)

# ================= PREDICTION =================
if st.button("🔍 Predict Risk"):
    user_df = pd.DataFrame(0, index=[0], columns=feature_names)

    user_df['age'] = age
    user_df['gender'] = 1 if gender == 'male' else 0
    user_df['smoker'] = 1 if smoker == 'yes' else 0
    user_df['smoking_years'] = smoking_years
    user_df['cigarettes_per_day'] = cigarettes
    user_df['passive_smoking'] = 1 if passive == 'yes' else 0
    user_df['bmi'] = bmi
    user_df['air_pollution_index'] = air
    user_df['family_history_cancer'] = 1 if family == 'yes' else 0
    user_df['asthma'] = 1 if asthma == 'yes' else 0
    user_df['previous_tb'] = 1 if tb == 'yes' else 0
    user_df['chronic_cough'] = 1 if cough == 'yes' else 0
    user_df['chest_pain'] = 1 if chest_pain == 'yes' else 0
    user_df['oxygen_saturation'] = oxygen_saturation
    user_df['shortness_of_breath'] = 1 if breath == 'yes' else 0
    user_df['fatigue'] = 1 if fatigue == 'yes' else 0
    user_df['xray_abnormal'] = 1 if xray_abnormal == 'yes' else 0
    user_df['exercise_hours_per_week'] = exercise
    user_df['diet_quality'] = diet
    user_df['alcohol_units_per_week'] = alcohol
    user_df['healthcare_access'] = healthcare

    proba = model.predict_proba(user_df)[0]
    risk_percent = proba[1] * 100

    if risk_percent > 70:
        st.error(f"🚨 HIGH RISK: {risk_percent:.2f}%")
    elif risk_percent > 30:
        st.warning(f"⚠️ MODERATE RISK: {risk_percent:.2f}%")
    else:
        st.success(f"✅ LOW RISK: {risk_percent:.2f}%")
