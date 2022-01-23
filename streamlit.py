import streamlit as st
import pandas as pd


st.write("""# Loan Prediction
        ## Business Problem
        Predicted the Home credit's client's repayment abilities :
        Given the data of a client we have to predict if he/she is able to repay loan or will have difficulty in paying back.
        PS : Home Credit is an international non-bank financial institution
         """)
#import pickle
st.write("""## Test Client's data :""")

test_client = pd.read_csv("test_streamlit_2.csv")
#X_test= pd.read_pickle("test_data.pkl")
#X_test
test_client

ID_client = test_client['SK_ID_CURR'].unique()

st.write("""## Select the SK_ID_CURR :""")
#SK_ID_CURR = st.sidebar.selectbox("Select a client:",ID_client)
SK_ID_CURR = st.selectbox("Select a client:",ID_client)


st.write("""## Client profile :""")
client_profile = test_client[test_client['SK_ID_CURR'] == SK_ID_CURR]
client_profile

#Prediction's test :
import pickle

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
    
prediction = loaded_model.predict_proba(client_profile.loc[:,'SK_ID_CURR':])
st.write("""## Prediction's result""")
prediction

st.write("""## Bank decision :""")
if(prediction[0][0]>0.5):
    st.write("""##### Bankrupt customer""")
    st.write("""##### Loan Denied""")
else:
    st.write("""##### Solvent customer""")
    st.write("""##### Loan Granted""")

st.write("""## Top features that helped in prediction""")
st.write("""These are the values of top 5 features used in prediction

""")
client_features_selected = client_profile[['SK_ID_CURR','EXT_SOURCE_3','CODE_GENDER_F','CODE_GENDER_M','PAYMENT_DIFF_max','FLAG_DOCUMENT_3','EXT_SOURCE_1']]
client_features_selected

st.write("""## Distribution of Top features""")
# plot features
data_client = pd.read_csv("test_streamlit_3.csv")

import seaborn as sns
import matplotlib.pyplot as plt

#plt.style.use('fivethirtyeight')
#plt.figure(figsize = (10, 8))
st.write("""##### EXT_SOURCE_3""")

sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'EXT_SOURCE_3'], label = 'target == 0' )
sns.kdeplot(data_client.loc[data_client['TARGET'] == 1, 'EXT_SOURCE_3'], label = 'target == 1' )

#plt.legend()
#st.line_chart(data_client.loc[data_client['TARGET'] == 0, 'EXT_SOURCE_3'])
#st.write(sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'EXT_SOURCE_3'], label = 'target == 0' ))
st.pyplot()

st.write("""##### CODE_GENDER_F""")
sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'CODE_GENDER_F'], label = 'target == 0' )
sns.kdeplot(data_client.loc[data_client['TARGET'] == 1, 'CODE_GENDER_F'], label = 'target == 1' )
st.pyplot()

st.write("""##### CODE_GENDER_M""")
sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'CODE_GENDER_M'], label = 'target == 0' )
sns.kdeplot(data_client.loc[data_client['TARGET'] == 1, 'CODE_GENDER_M'], label = 'target == 1' )
st.pyplot()

st.write("""##### PAYMENT_DIFF_max""")
sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'PAYMENT_DIFF_max'], label = 'target == 0' )
sns.kdeplot(data_client.loc[data_client['TARGET'] == 1, 'PAYMENT_DIFF_max'], label = 'target == 1' )
st.pyplot()

st.write("""##### FLAG_DOCUMENT_3""")
sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'FLAG_DOCUMENT_3'], label = 'target == 0' )
sns.kdeplot(data_client.loc[data_client['TARGET'] == 1, 'FLAG_DOCUMENT_3'], label = 'target == 1' )
st.pyplot()

st.write("""##### EXT_SOURCE_1""")
sns.kdeplot(data_client.loc[data_client['TARGET'] == 0, 'EXT_SOURCE_1'], label = 'target == 0' )
sns.kdeplot(data_client.loc[data_client['TARGET'] == 1, 'EXT_SOURCE_1'], label = 'target == 1' )
st.pyplot()
