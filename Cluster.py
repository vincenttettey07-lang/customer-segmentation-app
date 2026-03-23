import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# setting page configuration
st.set_page_config(
    page_title='Customer Segmentation',
    layout='centered',
)

st.title('Customer Segmentation')
st.write("Created by Vincent Tettey")
st.markdown("---")

st.write("🛃 Please provide the following details about customer")
# Loading the model and the scaler 
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('Scaler.pkl')

st.markdown(
    """
    <style>
    div.stButton > button {
    background-color: orange;
    color:blue;
    font-weight:bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    border: none;
    animation: pulse 2s infinite;
    transition: 0.3s ease-in-out; 
    }
    @keyframes pulse{
    0% {box-shadow: 0 0 5px #16A34A;
        }
        50% {box-shadow: 0 0 2px
    #16A34A, 0 0 40px #16A34A;    }
        100% { box-shadow: 0 0 5px
    #16A34A;    }
    }
    div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px #16A34A, 0 0 50px #16A34A;
    }
    </style>
    """,
    unsafe_allow_html=True
) 



# Inputs
recency = st.number_input("Recency (Days since last purchase)", min_value =0)
frequency = st.number_input("Frequency (Number of purchases)", min_value=0)
monetary = st.number_input("Monetary Value (Total Amount Spent in $)", min_value=0.00)

# Labels
cluster_labels = {0:'Low Value',1: 'Lost High Value',2: 'Churned',3:'VIP'}

st.markdown("---")

#Prediction
if st.button("Predict Customer Segment"):
    with st.spinner("Analysing customer data..."):
        import time
        time.sleep(3)
    features = np.array([[recency,frequency,monetary]])

    features_scaled = scaler.transform(features)
    segment_info = {
        "VIP": "These are your most valuable customers. They buy frequently.",
        "Low Value": "Customers with low spending and low engagement.",
        "Lost High Value": "These customers used to buy but are becoming inactive.",
        "Churned": "These customers have not purchased for a long time.",
        "Not a customer yet": "No transaction history yet."
        
    }

    recommendations = {
        "VIP": "Reward them with loyalty bonuses and exclusive offers.",
        "Low Value": "Offer incentives to increase engagement", 
        "Lost High Value": "Send personalized offers to re-engage them.",
        "Churned": "Win them back with strong promotions.",
        "Not a customer yet": "Encourage first purchase with welcome offers."
    }
    st.markdown("---")
    
    if recency == 0 and frequency == 0 :
        segment = 'Not a customer yet'
        st.success(f'Customer Segment is :  {segment}')

    elif recency != 0 and frequency == 0 and monetary != 0:
        segment = 'Not a customer yet'
        st.success(f'Customer Segment is :  {segment}')

    elif recency != 0 and frequency == 0 and monetary == 0:
        segment = 'Not a customer yet'
        st.success(f'Customer Segment is :  {segment}')
        
    elif recency >= 365:
        segment = 'Churned'
        st.error(f'Customer Segment is : 😓 {segment}')

    elif monetary < 100 and recency > 180:
        segment = 'Churned'
        st.error(f'Customer Segment is : 😓 {segment}')

    elif recency != 0 and frequency != 0 and monetary  < 40:
        segment = 'Low Value'
        st.error(f'Customer Segment is : 😓 {segment}')

    elif recency != 0 and frequency != 0 and monetary == 0:
        segment = 'Not a customer yet'
        st.success(f'Customer Segment is :  {segment}')
        
    elif recency == 0 and frequency != 0 and monetary == 0.00:
        segment = 'Not a customer yet'
        st.success(f'Customer Segment is :  {segment}')


    
    else:
        cluster = model.predict(features_scaled)
        segment = cluster_labels[cluster[0]]
        if segment == 'VIP':
                st.success(f'Customer Segment is : 🤑 {segment}')
        else:
            st.error(f'Customer Segment is : 😓 {segment}')
    
    st.info(f'💡 Insight: {segment_info[segment]}') 
    st.info(f"📌Recommendation: {recommendations[segment]}")
    st.markdown("---")
    st.header("Customer history summary")
    bar_chart = pd.DataFrame({'RFM':['Recency','Frequency','Monetary'],'Inputs':[recency, frequency, monetary]})
    st.bar_chart(bar_chart.set_index('RFM'))

   

   




