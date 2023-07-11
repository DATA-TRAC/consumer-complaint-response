#standard imports
import pandas as pd 

#get/create files
import os
from google.oauth2 import service_account
import pandas_gbq

#splitting
from sklearn.model_selection import train_test_split

#vizzes
import matplotlib.pyplot as plt
import seaborn as sns

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

print('imports loaded successfully, awaiting commands...')

#------------------------------------------------------------- ACQUIRE -------------------------------------------------------------
    
def check_file_exists_gbq(csv_fn, json_fn):
    '''
    check if file exists in my local directory, if not, pull from google big query db
    using service key token, return dataframe and save to a csv in the local directory
    '''
    #credentials
    credentials = service_account.Credentials.from_service_account_file(
    [json_fn],)
    
    #if/else file exists in local directory
    if os.path.isfile(csv_fn):
        print('csv file found and loaded')
        return pd.read_csv(csv_fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pandas_gbq.read_gbq("select * from cfpb_complaints.complaint_database",dialect='standard',project_id="my-ds-projects",use_bqstorage_api=True,credentials=credentials)
        df.to_csv(csv_fn)
        return df
    
#------------------------------------------------------------- PREPARE -------------------------------------------------------------



#------------------------------------------------------------- VIZZES -------------------------------------------------------------

