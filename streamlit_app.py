import streamlit as st
import pandas as pd
import pickle
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit.components.v1 as components
from IPython.display import HTML

test_client= pd.read_pickle("test2_data.pkl")
data_client = pd.read_csv("test_streamlit_4.csv") #contient target
list_lime = joblib.load("list_lime.pkl")



url = "https://loanpredictionoc.herokuapp.com/predict"

st.write("""# Loan Prediction
        ## Business Problem
        Predicted the Home credit's client's repayment abilities :
        Given the data of a client we have to predict if he/she is able to repay loan or will have difficulty in paying back.
        PS : Home Credit is an international non-bank financial institution
         """)

st.write("""## Test Client's data :""")

test_client
#st.write(str(len(test_client)))

ID_client = test_client.index

st.write("""## Select a client :""")
index_client = st.selectbox("Select a client:",list(ID_client))

if index_client !=None:
    

    st.write("""## Client profile :""")
    client_profile = test_client[test_client.index == index_client]
    client_profile

    headers = {"Content-Type": "application/json"}
    requette = requests.request(method='POST', headers=headers, url=url, json={"index":index_client})
    
    reponse = requette.json()
    prediction=reponse["prediction"]
    probability=reponse["probability"]
    st.write("""## Loan Prediction :""")
    st.write("probability:",str(probability))
    st.write("prediction:",prediction)
    
    st.write("""## Local interpretability :""")
    interpretabilite=list_lime[index_client]
    components.html(interpretabilite.as_html(),height=700)
    
    st.write("""## Top features that helped in prediction""")
    st.write("""These are the values of top 5 features used in prediction
    #""") 
    client_features_selected = client_profile[['EXT_SOURCE_2','EXT_SOURCE_3','DPD_max','CODE_GENDER_F','CODE_GENDER_M','PAYMENT_RATE','PAYMENT_DIFF_max']]
    #,'FLAG_DOCUMENT_3','EXT_SOURCE_1'
    client_features_selected

#st.write("""## Distribution of Top features""")
# plot features



    plt.style.use('fivethirtyeight')
    plt.figure(figsize = (10, 8))
    st.write("""##### EXT_SOURCE_2""")

    ax1=sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'EXT_SOURCE_2'], label = 'target == 0' )
    ax2=sns.kdeplot(data_client.loc[data_client['TARGET'] == 1, 'EXT_SOURCE_2'], label = 'target == 1' )
    plt.axvline(x=data_client.iloc[index_client]["EXT_SOURCE_2"],color="red",ls="--",lw=2.5)
    plt.legend()
    st.pyplot()

    st.write("""##### EXT_SOURCE_3""")
    ax1=sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'EXT_SOURCE_3'], label = 'target == 0' )
    ax2=sns.kdeplot(data_client.loc[data_client['TARGET'] == 1, 'EXT_SOURCE_3'], label = 'target == 1' )
    plt.axvline(x=data_client.iloc[index_client]["EXT_SOURCE_3"],color="red",ls="--",lw=2.5)
    plt.legend()
    st.pyplot()

    st.write("""##### CREDIT_TYPE_Car loan_mean""")
    ax1=sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'CREDIT_TYPE_Car loan_mean'], label = 'target == 0' )
    ax1=sns.kdeplot(data_client.loc[data_client['TARGET'] == 1, 'CREDIT_TYPE_Car loan_mean'], label = 'target == 1' )
    plt.axvline(x=data_client.iloc[index_client]["CREDIT_TYPE_Car loan_mean"],color="red",ls="--",lw=2.5)
    plt.legend()
    st.pyplot()
