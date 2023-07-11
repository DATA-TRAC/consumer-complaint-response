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

def clean_data(df):
    """
    The `clean_data` function takes a DataFrame as input, performs various data cleaning operations such
    as changing column types, deriving a new column, replacing missing values, dropping rows, and
    dropping unnecessary columns, and returns the cleaned DataFrame.
    
    :param df: The parameter `df` is a pandas DataFrame that contains the data to be cleaned
    :return: a cleaned dataframe with certain columns dropped and missing values replaced.
    """
    # Change column type to datetime64[ns] for column: 'date_received'
    df = df.astype({'date_received': 'datetime64[ns]'})
    # Derive column 'narrative' from columns: 'consumer_complaint_narrative', 'issue', 'subissue'
    df.insert(5,'narrative',df.apply(
            lambda row: f'{str(row.issue)} {str(row.subissue)} {str(row.consumer_complaint_narrative)}',
            axis=1,),)
    # Replace missing values with "UNKNOWN" in column: 'state' and "Average Person" in column: 'tags'
    df = df.fillna({'state': "UNKNOWN",'tags': "Average Person"})
    # Drop rows with missing data in column: 'company_response_to_consumer'
    df = df.dropna(subset=['company_response_to_consumer'])
    # Filter rows based on column: 'company_response_to_consumer'
    df = df[df['company_response_to_consumer'] != "In progress"]
    # drop columns not used for explore or modeling
    return df.drop(
        columns=[
            'subproduct',
            'complaint_id',
            'consumer_disputed',
            'timely_response',
            'date_sent_to_company',
            'submitted_via',
            'company_public_response',
            'zip_code',
            'consumer_consent_provided',
            'issue',
            'subissue',
            'consumer_complaint_narrative',
        ]
    )

#------------------------------------------------------------- VIZZES -------------------------------------------------------------

