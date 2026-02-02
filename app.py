import streamlit as st
import pandas as pd 
import joblib

# ===============================
# Ownership Badge in Sidebar
# ===============================
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 12px; 
                background: linear-gradient(135deg, #667eea 0%,#764ba2 100%); 
                border-radius: 10px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
        <span style="font-size: 11px; color: rgba(255,255,255,0.85); letter-spacing: 1px;">CREATED BY</span><br>
        <a href="https://www.linkedin.com/in/eddine-nasri-71a5b2160/" target="_blank" 
           style="font-size: 15px; font-weight: 600; color: #ffffff; text-decoration: none;">
            Eddine Nasri
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# Project Description in Sidebar
# ===============================
st.sidebar.markdown(
    """



    **Credit Risk Prediction App**  

    -Predicts applicant credit risk using the **best-performing model selected from multiple AI/ML candidates**, trained on the **German credit dataset (large, real-world dataset)**.  
   
    -Includes **feature encoding, preprocessing, and model evaluation** to ensure robust and reliable predictions.  
    
    -Demonstrates **applied AI/ML in finance** with an interactive app.
    """,
)



model = joblib.load("random_forest_credit_model.pkl")
encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Saving accounts","Housing","Checking account"]}

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad")

age = st.number_input("Age", min_value=18, max_value=80,value=30)
sex = st.selectbox("Sex",["male","female"])
job = st.number_input("Job (0-3)", min_value = 0, max_value=3, value =1)
housing = st.selectbox("Housing",["own","rent","free"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich"])
checking_account = st.selectbox("Checking Accounts", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit amount", min_value=1, value=1000 )
duration = st.number_input("Duration (months)", min_value=1, value=12)

input_df = pd.DataFrame({
    "Age" : [age],
    "Sex" : [encoders["Sex"].transform([sex])[0]],
    "Job" : [job],
    "Housing" : [encoders["Housing"].transform([housing])[0]],
    "Saving accounts" : [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account" : [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration" : [duration]
})

if st.button("Predict Risk"):
    pred= model.predict(input_df)[0]

    if pred ==1:
        st.success("The predicted credit risk is : **GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")

# 1 is good Lower risk 0 is bad higher risk
