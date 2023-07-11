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

#prepare imports
import re
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

#nltk sentiment
from nltk.sentiment import SentimentIntensityAnalyzer

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
    json_fn,)
    
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

def lower_everything(string):
    return string.str.lower()

def normalize_everything(string):
    string = unicodedata.normalize('NFKD', string).encode('ascii','ignore').decode('utf-8')
    return string

def specials_removed(string):
    string = re.sub(r'[^a-z0-9\'\s]', '', string)
    return string

def basic_clean(string):
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii','ignore').decode('utf-8')
    string = re.sub(r'[^a-z0-9\'\s]', '', string)
    return string

def token_it_up(string):
    tokenize = nltk.tokenize.ToktokTokenizer()
    string = tokenize.tokenize(string, return_str=True)
    return string

def stemmer(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string = ' '.join(stems)
    return string

def lemmad(string):
    wnl = nltk.stem.WordNetLemmatizer()
    string = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(string)
    return string

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    sls = stopwords.words('english')
    
    sls = set(sls) - set(exclude_words)
    sls = sls.union(set(extra_words))
    
    words = string.split()
    filtered = [word for word in words if word not in sls]
    string = ' '.join(filtered)
    return string

def clean_df(df, exclude_words=[], extra_words=[]):
    '''
    send in df with columns: title and original,
    returns df with original, clean, stemmed, and lemmatized data
    '''
    df['clean'] = df.original.apply(basic_clean).apply(token_it_up).apply(remove_stopwords)
    df['stem'] = df.clean.apply(stemmer)
    df['lemma'] = df.clean.apply(lemmad)
    
    return df


def clean(text):
    '''
    A simple function to cleanup text data.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        list: A list of lemmatized words after cleaning.
    '''
    #assigning additional stopwords
    ADDITIONAL_STOPWORDS = ['r', 'u', '2', '4', 'ltgt']
    
    # basic_clean() function from last lesson:
    # Normalize text by removing diacritics, encoding to ASCII, decoding to UTF-8, and converting to lowercase
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore') #most frequently used for base text creation - works great with SQL
             .lower())
    
    # Remove punctuation, split text into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    
    # lemmatize() function from last lesson:
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Combine standard English stopwords with additional stopwords
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words

#------------------------------------------------------------- NLTK -------------------------------------------------------------

def sentiment_analysis(df):
    '''
    this function applies NLTK sentiment ananlysis to each narrative in the df
    '''
    # Initialize the sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Apply the sentiment intensity analyzer to the 'consumer_complaint_narrative' column
    df['sentiment'] = df['narrative'].apply(lambda complaint: sia.polarity_scores(complaint))