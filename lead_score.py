a=[]
a.append(0)

# %% 
import pandas as pd
import numpy as np
import glob
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import category_encoders as ce
import pickle
import requests
from google.cloud import bigquery
import datetime as dt  
import os
import warnings
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime,date
from google.cloud import datastore
warnings.simplefilter("ignore")

import sys

import time,  traceback
from datetime import datetime, timedelta
import pytz

# %%
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(location="US")
print("Client creating using default project: {}".format(client.project))
query = """
SELECT * FROM  `###` """

query_job = client.query(
    query,
    # Location must match that of the dataset(s) referenced in the query.
    location="US",
)  # API request - starts the query
final_df = query_job.to_dataframe()
print("row",final_df.shape)
### export csv
# final_df_convert.to_csv(train_folder+"convert.csv",index=False)

# %%
query_2 = """
    SELECT * FROM  `###` """
    
query_job_2 = client.query(
        query_2,
        location = "US",
    )
LMS_lead_data = query_job_2.to_dataframe()
print("row",LMS_lead_data.shape)

# %%
d1=LMS_lead_data[[ 'DATE_IST','###']]

# %%
try:
    unmapped_data_ga = pd.read_csv('###.csv')
    unmapped_data_lms = pd.read_csv('###.csv')
    final_df.append(unmapped_data_ga)
    d1.append(unmapped_data_lms)
except:
    print("Initial run")

# %%
final_df = final_df.sort_values(by = ['AUC_Code','date','visitStartTime'], ascending = False).drop_duplicates(subset = ['AUC_Code'],keep = 'first')
d1 = d1.sort_values(by = ['mx_AngelCode','DATE_IST'], ascending = False).drop_duplicates(subset = ['mx_AngelCode'],keep = 'first')
    
is_matching_ga = (final_df['AUC_Code'].isin(d1['mx_AngelCode'])) & (final_df.AUC_Code != 'None') & (~(final_df['AUC_Code'].isnull()))
is_matching_lms = (d1['mx_AngelCode'].isin(final_df['AUC_Code'])) & (d1.mx_AngelCode != 'None') & (~(d1['mx_AngelCode'].isnull()))

# %%
unmapped_ga = final_df.loc[~(is_matching_ga),:]
unmapped_ga = unmapped_ga.loc[(unmapped_ga['AUC_Code']!='None') & ~(unmapped_ga['AUC_Code'].isnull()),:]
unmapped_lms = d1.loc[~(is_matching_lms),:]
unmapped_lms = unmapped_lms.loc[(unmapped_lms['mx_AngelCode']!='None') & ~(unmapped_lms['mx_AngelCode'].isnull()),:]
    
unmapped_ga.to_csv('###.csv')
unmapped_lms.to_csv('###.csv')
    

# %%
final_df = final_df.loc[is_matching_ga,:]
d1 = d1.loc[is_matching_lms,:]

# %%
is_matching_ga.shape

# %%
is_matching_lms.shape

# %%
data=pd.merge(final_df,d1,how='inner',left_on='AUC_Code',right_on='mx_AngelCode')

# %%
data.shape

# %%
data1=data[['AUC_Code','hits','mobileDeviceMarketingName', 'city','mx_App_Status_Compare',
   'mx_Application_Source', 'mx_Application_Status',
   'mx_Application_Status_First_time_Drop_Off','mx_Date_of_Birth', 'LeadAge', 'mx_Lead_Medium','Source','mx_State','mx_Client_Code','date','visitStartTime','ProspectID'
   ]]

cols = ['Bengaluru', 'Indore', 'Lucknow', 'Mumbai', 'Hyderabad', 'Pune',
'Jaipur', 'Delhi', 'Ludhiana', 'Patna', 'Bhubaneswar', 'Chennai', 'Guwahati',
'Ahmedabad', 'Agra', 'Kolkata']

data1['city'].loc[data1['city'].apply(lambda x: (x not in cols) )] = 'Others'

cols = ['(not set)', 'Y15', 'A5', 'Y12', 'Galaxy J7 Prime', 'A5s', 'S1',
'Redmi 9', '6', 'Y11 (2019)', 'Galaxy M31', 'A3s', 'Y20', 'Y91C', 'C2',
'Galaxy A50', 'Y91']

data1['mobileDeviceMarketingName'].loc[data1['mobileDeviceMarketingName'].apply(lambda x: (x not in cols) )] = 'Others'


cols = ['SMS', 'google', '(direct)', 'facebook', 'deals102', 'MXTakaTak_Clip',
'brand_Mxtakatak', 'bing', 'YoAds', 'web', 'DICP_SMS']

data1['Source'].loc[data1['Source'].apply(lambda x: (x not in cols) )] = 'Others'

cols = ['Karnataka', 'Madhya Pradesh', 'Andhra Pradesh', 'Uttar Pradesh',
'Maharashtra', 'Gujarat', 'Telangana', 'Kerala', 'Rajasthan', 'Delhi', 'Punjab',
'Haryana', 'Bihar', 'Odisha', 'Tamil Nadu', 'Assam', 'West Bengal']

data1['mx_State'].loc[data1['mx_State'].apply(lambda x: (x not in cols) )] = 'Others'

cols = ['Web',
'null',
'other',
'SS-GoogleSearch',
'Vendor - Mailer',
'SMS',
'RnE – Tele',
'Online – Tele',
'ABMA',
'Earn_Rewards',
'RnE - Client',
'Earn Rewards',
'rne_client1',
'DRA - B2C',
'SS- Facebook',
'Discovery',
'SS-Innovation',
'DRA - Youtube',
'ReferNEarn',
'Brand Innovation',
'RnE-Client'
]

data1['mx_Lead_Medium'].loc[data1['mx_Lead_Medium'].apply(lambda x: (x not in cols) )] = 'ABMA'

data1['mx_Date_of_Birth']=pd.to_datetime(data1['mx_Date_of_Birth'],format="%Y-%m-%d %H:%M:%S.%f")

# %%
def age(born):
    today = date.today()
    return today.year - born.year - ((today.month, 
                                           today.day) < (born.month, 
                                                         born.day))


data1['Age'] = data1['mx_Date_of_Birth'].apply(age)
data1['Age']=pd.to_numeric(data1['Age'])
data1.drop(labels=['mx_Date_of_Birth'],axis=1,inplace=True)

bins=[0,22,40,60,100]
labels=['Teenage','Adult','Senior','Elder']
data1['Age_Group']=pd.cut(data1['Age'],bins=bins,labels=labels,right=False,ordered=False)

data1.drop(labels=['Age'],axis=1,inplace=True)

data1['mx_App_Status_Compare']=data1['mx_App_Status_Compare'].fillna("No Values")

data1['mx_Application_Status']=data1['mx_Application_Status'].fillna("No Values")

data1['mx_Application_Status_First_time_Drop_Off']=data1['mx_Application_Status_First_time_Drop_Off'].fillna("No Values")

data1['LeadAge']=data1['LeadAge'].fillna(-1)

data1['mx_Application_Source']=data1['mx_Application_Source'].fillna("No Values")

categories = np.array(
 ['Teenage','Adult','Senior','Elder','No Values'])
data1['Age_Group'] = pd.Categorical(
data1['Age_Group'], categories=categories, ordered=False)

data1['Age_Group']=data1['Age_Group'].fillna("No Values")



scaler = MinMaxScaler()
scaler.fit(data1[['hits','LeadAge']])
data1[['hits','LeadAge']] = scaler.transform(data1[['hits','LeadAge']])

# %%
cat_vars=[ 'mobileDeviceMarketingName', 'city',
       'mx_App_Status_Compare', 'mx_Application_Source',
       'mx_Application_Status', 'mx_Application_Status_First_time_Drop_Off'
       , 'mx_Lead_Medium', 'Source', 'mx_State','Age_Group']

onehotencoder = joblib.load('/home/jupyter/Lead_Scoring_App/transformers/sklearn_ohe.pkl')
sklearn_ohe = onehotencoder.transform(data1[cat_vars])

#col = []
#for i,z in zip(onehotencoder.categories_,cat_vars):
#    col.append(i + '_' + z)
#col = [item for sublist in col for item in sublist]
#print(col)

data1= pd.concat([data1,sklearn_ohe],axis = 1)
cat_vars=['mobileDeviceMarketingName', 'city',
       'mx_App_Status_Compare', 'mx_Application_Source',
       'mx_Application_Status', 'mx_Application_Status_First_time_Drop_Off'
       , 'mx_Lead_Medium', 'Source', 'mx_State','Age_Group']
data_vars=data1.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

# %%
data_final=data1[to_keep]

# %%
prediction_df = data_final[['AUC_Code','date','visitStartTime','ProspectID']]



# Drop duplicate rows
data_final = data_final.drop_duplicates(keep='first').reset_index(drop= True)

X = [x for x in data_final.columns.to_list() if x not in ['AUC_Code','date','visitStartTime','mx_Client_Code','ProspectID']]

X = data_final[X]

# %%
import pickle
filename = '/home/jupyter/Lead_Scoring_App/AppModel/model1.sav'
clf = pickle.load(open(filename, 'rb'))


# In[32]:


y_prd= clf.predict_proba(X.values)


# In[33]:


unique_elements, counts_elements = np.unique(clf.predict(X), return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


# In[34]:


prediction_df['lead_score'] = y_prd[:,1]




fmt = "%Y-%m-%d %H:%M:%S"
now_utc = datetime.now(pytz.timezone('UTC'))


# In[36]:


prediction_datetime = (now_utc.astimezone(pytz.timezone('Asia/Kolkata'))- timedelta(minutes=30)).strftime(fmt)
print("date of prediction --",prediction_datetime)


# In[37]:


# datetime.strftime("%Y%m%d",prediction_datetime)
prediction_table_date =  (now_utc.astimezone(pytz.timezone('Asia/Kolkata'))- timedelta(minutes=30)).strftime("%Y%m%d")


#lower_thresh = prediction_df['lead_score'].quantile(0.3)
#higher_thresh = prediction_df['lead_score'].quantile(0.7)

lower_thresh = 0.03
higher_thresh = 0.15

# In[39]:


print ("Lower_threshold : {}".format(lower_thresh))
print ("higher_thresh : {}".format(higher_thresh))


# In[40]:


prediction_df['bucket'] =  np.where(prediction_df['lead_score'] >= higher_thresh , 1, 0)
prediction_df['bucket'] =  np.where( ((prediction_df['lead_score'] >= lower_thresh ) & (prediction_df['lead_score'] < higher_thresh))
                           , 2, prediction_df['bucket'])
prediction_df['bucket'] =  np.where( prediction_df['lead_score'] <= lower_thresh , 3, prediction_df['bucket'])

prediction_df['time'] = prediction_datetime


# In[41]:

prediction_df["AUC_Code"] = prediction_df["AUC_Code"].apply(str)
prediction_df["date"] = prediction_df["date"].apply(str)
prediction_df["time"] = prediction_df["time"].apply(str)
prediction_df["ProspectID"] = prediction_df["ProspectID"].apply(str)

unique_ga_count = prediction_df['AUC_Code'].nunique()

# %%
print ("Data to be insert : {}".format(prediction_df.shape))

# In[44]:

print ("Adding data to BQ :app_lead_score table")

client = bigquery.Client()

# Production table
try : 
    query = """
    SELECT * FROM `angel-broking-tvc.daily_serving_app_lead_scoring.app_lead_score_{0}`  
    """.format(prediction_table_date)


    query_job = client.query(
        query
    )  

    bq_df_exclude = query_job.to_dataframe()
except : 

    bq_df_exclude = pd.DataFrame(columns = ['AUC_Code', 'date', 'visitStartTime', 'ProspectID', 'lead_score', 'bucket',
       'time'])


# In[45]:


prediction_df_bg = prediction_df.sort_values('lead_score', ascending=False)
prediction_df_bg = prediction_df_bg.drop_duplicates(subset=['AUC_Code','date','visitStartTime'], keep='first').reset_index(drop=True)


# In[46]:


if not bq_df_exclude.empty:
    prediction_df_bg = pd.concat([prediction_df_bg, bq_df_exclude]).reset_index(drop=True)
    prediction_df_bg = prediction_df_bg.drop_duplicates(subset=['AUC_Code','date','visitStartTime'],keep=False).reset_index(drop=True)
else:
    print ("Empty app_lead_score BQ table")
    pass


prediction_df_bg = prediction_df_bg.loc[prediction_df_bg['time'] == prediction_datetime]
print("Data insert to app_lead_score table ---",prediction_df_bg.shape)
print("Bigquery intertion start")


# In[48]:


prediction_df_bg["AUC_Code"] = prediction_df_bg["AUC_Code"].apply(str)
prediction_df_bg["date"] = prediction_df_bg["date"].apply(str)
prediction_df_bg["time"] = prediction_df_bg["time"].apply(str)
prediction_df["ProspectID"] = prediction_df["ProspectID"].apply(str)


# %%
if not prediction_df_bg.empty:

    bq_client = bigquery.Client()
    # change here
    dataset_ref = bq_client.dataset('daily_serving_app_lead_scoring')
    job_config = bigquery.LoadJobConfig(schema=[
                bigquery.SchemaField(name="AUC_Code", field_type="STRING"),
                bigquery.SchemaField(name="date", field_type="STRING"),
                bigquery.SchemaField(name="visitStartTime", field_type="INTEGER"),
                bigquery.SchemaField(name="ProspectID", field_type="STRING"),
                bigquery.SchemaField(name="lead_score", field_type="FLOAT"),
                bigquery.SchemaField(name="bucket", field_type="INTEGER"),
                bigquery.SchemaField(name="time", field_type="STRING")
            ])
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
    # change here
    table_ref = dataset_ref.table("app_lead_score_"+prediction_table_date)
    dataframe = pd.DataFrame(prediction_df_bg.to_records(index=False))
    job = bq_client.load_table_from_dataframe(dataframe, table_ref, job_config=job_config)
    job.result() # Waits for table load to complete.

else:
    pass
prediction_df_bg['bucket_type'] = prediction_df_bg['bucket'].replace({1:"High", 2:"Medium", 3:"Low"})


#count_remove_duplicate = count_of_unique_user(prediction_df, prediction_datetime, prediction_table_date)

print ("BigQuery Insertion end ----------------")

print ("Store audience to BQ LMS_pending_audience table")

if not prediction_df_bg.empty:

    bq_client = bigquery.Client()
    # change here
    dataset_ref = bq_client.dataset('daily_serving_ab_lead_scoring')
    job_config = bigquery.LoadJobConfig(schema=[
                bigquery.SchemaField(name="ProspectID", field_type="STRING"),
                bigquery.SchemaField(name="visitStartTime", field_type="INTEGER"),
                bigquery.SchemaField(name="bucket", field_type="INTEGER"),
                bigquery.SchemaField(name="lead_score", field_type="FLOAT")
            ])
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
    # change here
    table_ref = dataset_ref.table("LMS_pending_audience")
    dataframe = pd.DataFrame(prediction_df_bg[["ProspectID", "visitStartTime", "bucket", "lead_score"]].to_records(index=False))
    job = bq_client.load_table_from_dataframe(dataframe, table_ref, job_config=job_config)
    job.result() # Waits for table load to complete.

else:
    pass
    